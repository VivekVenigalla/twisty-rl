;;; twisty-paint.el --- Enter a physical puzzle and step through its solution  -*- lexical-binding: t; -*-

;; Package-Requires: ((emacs "28.1"))
;; Keywords: games

;;; Commentary:

;; `twisty-paint' shows the unfolded net of the current puzzle.  Type a colour key to paint
;; the sticker under the cursor (the cursor then advances in reading order, so a face is nine
;; key presses), RET checks the colouring.  The daemon's Schreier-Sims engine either explains
;; what is impossible about it ("one edge is flipped", ...) or produces a solution, which
;; `twisty-steps-mode' lets you walk through move by move while the 3D puzzle follows.

;;; Code:

(require 'cl-lib)
(require 'subr-x)
(require 'twisty-dashboard)

(defconst twisty-paint-buffer-name "*twisty-paint*")
(defconst twisty-steps-buffer-name "*twisty-solution*")

(defvar-local twisty-paint--layout nil "Layout alist received from the daemon.")
(defvar-local twisty-paint--colors nil "Vector: colour index of every facet.")
(defvar-local twisty-paint--current 0 "Facet under the cursor.")
(defvar-local twisty-paint--order nil "Editable facets in typing order.")
(defvar-local twisty-paint--message "" "Verdict of the last check.")

(defun twisty-paint--get (key)
  "Value of KEY in the layout."
  (alist-get key twisty-paint--layout))

(defun twisty-paint--cell (facet)
  "Grid cell (ROW COL) of FACET."
  (nth facet (twisty-paint--get 'cells)))

(defun twisty-paint--fixed-p (facet)
  "Non-nil when FACET never moves (a centre) and cannot be painted."
  (nth facet (twisty-paint--get 'fixed)))

(defun twisty-paint--contrast (hex)
  "Black or white, whichever reads better on colour HEX."
  (let ((rgb (color-name-to-rgb hex)))
    (if (and rgb (> (+ (* 0.299 (nth 0 rgb)) (* 0.587 (nth 1 rgb)) (* 0.114 (nth 2 rgb))) 0.55))
        "#000000" "#ffffff")))

(defun twisty-paint--typing-order ()
  "Editable facets sorted face by face, then in reading order."
  (let* ((home (twisty-paint--get 'solved))
         (facets (cl-remove-if #'twisty-paint--fixed-p (number-sequence 0 (1- (length home))))))
    (sort facets
          (lambda (a b)
            (let ((ka (list (nth a home) (car (twisty-paint--cell a)) (cadr (twisty-paint--cell a))))
                  (kb (list (nth b home) (car (twisty-paint--cell b)) (cadr (twisty-paint--cell b)))))
              (or (< (nth 0 ka) (nth 0 kb))
                  (and (= (nth 0 ka) (nth 0 kb))
                       (or (< (nth 1 ka) (nth 1 kb))
                           (and (= (nth 1 ka) (nth 1 kb)) (< (nth 2 ka) (nth 2 kb)))))))))))

(defun twisty-paint--draw ()
  "Redraw the paint buffer."
  (let* ((inhibit-read-only t)
         (rows (twisty-paint--get 'rows)) (cols (twisty-paint--get 'cols))
         (palette (twisty-paint--get 'palette)) (names (twisty-paint--get 'names))
         (keys (twisty-paint--get 'keys)) (need (twisty-paint--get 'need))
         (grid (make-vector (* rows cols) nil))
         (bg (or (face-background 'default nil t) "#000000"))
         (have (make-vector (length palette) 0))
         (target nil))
    (cl-loop for cell in (twisty-paint--get 'cells) for facet from 0
             do (aset grid (+ (* (car cell) cols) (cadr cell)) facet))
    (cl-loop for c across twisty-paint--colors do (aset have c (1+ (aref have c))))
    (erase-buffer)
    (insert (propertize (format "Paint your %s" (twisty-paint--get 'puzzle)) 'face 'twisty-title) "\n"
            (propertize "Hold the puzzle so its centres match the lettered cells, then type the colours you see."
                        'face 'twisty-dim)
            "\n\n")
    (dotimes (r rows)
      (insert "  ")
      (dotimes (c cols)
        (let ((facet (aref grid (+ (* r cols) c))))
          (if (null facet)
              (insert "   ")
            (let* ((hex (nth (aref twisty-paint--colors facet) palette))
                   (here (= facet twisty-paint--current))
                   (text (cond (here "[]")
                               ((twisty-paint--fixed-p facet)
                                (truncate-string-to-width
                                 (nth (aref twisty-paint--colors facet) names) 2 nil ?\s))
                               (t "▁▁")))
                   (fg (if (or here (twisty-paint--fixed-p facet)) (twisty-paint--contrast hex) bg)))
              (when here (setq target (point)))
              (insert (propertize text 'face (list :foreground fg :background hex :weight 'bold)
                                  'twisty-facet facet)
                      " ")))))
      (insert "\n"))
    (insert "\n  ")
    (cl-loop for key in keys for hex in palette for name in names for i from 0
             do (insert (propertize (format " %s " key)
                                    'face (list :foreground (twisty-paint--contrast hex)
                                                :background hex :weight 'bold))
                        (propertize (format " %s %d/%d   " name (aref have i) (nth i need))
                                    'face (if (= (aref have i) (nth i need)) 'twisty-dim 'warning))))
    (insert "\n\n")
    (unless (string-empty-p twisty-paint--message)
      (insert "  " (propertize twisty-paint--message 'face 'warning) "\n\n"))
    (insert (propertize "  colour keys paint and advance · arrows move · DEL back · TAB next face\n"
                        'face 'twisty-dim)
            (propertize "  RET check and solve · C-c C-r start from solved · C-c C-l load the on-screen puzzle · C-c C-k quit"
                        'face 'twisty-dim))
    (goto-char (or target (point-min)))))

(defun twisty-paint--move (drow dcol)
  "Move the cursor to the nearest sticker in direction (DROW DCOL)."
  (let* ((here (twisty-paint--cell twisty-paint--current))
         (best nil) (best-score nil))
    (cl-loop for cell in (twisty-paint--get 'cells) for facet from 0
             do (let* ((dr (- (car cell) (car here))) (dc (- (cadr cell) (cadr here)))
                       (along (+ (* dr drow) (* dc dcol)))
                       (across (abs (- (* dr dcol) (* dc drow)))))
                  (when (and (> along 0) (not (twisty-paint--fixed-p facet)))
                    (let ((score (+ along (* 3 across))))
                      (when (or (null best-score) (< score best-score))
                        (setq best facet best-score score))))))
    (when best
      (setq twisty-paint--current best))
    (twisty-paint--draw)))

(defun twisty-paint-left () "Move one sticker left." (interactive) (twisty-paint--move 0 -1))
(defun twisty-paint-right () "Move one sticker right." (interactive) (twisty-paint--move 0 1))
(defun twisty-paint-up () "Move one sticker up." (interactive) (twisty-paint--move -1 0))
(defun twisty-paint-down () "Move one sticker down." (interactive) (twisty-paint--move 1 0))

(defun twisty-paint--step (n)
  "Move N stickers along the typing order (wrapping)."
  (let* ((pos (or (cl-position twisty-paint--current twisty-paint--order) 0))
         (len (length twisty-paint--order)))
    (when (> len 0)
      (setq twisty-paint--current (nth (mod (+ pos n) len) twisty-paint--order)))))

(defun twisty-paint-back ()
  "Move to the previous sticker."
  (interactive)
  (twisty-paint--step -1)
  (twisty-paint--draw))

(defun twisty-paint-next-face ()
  "Jump to the first sticker of the next face."
  (interactive)
  (let* ((home (twisty-paint--get 'solved))
         (face (nth twisty-paint--current home))
         (tail (cdr (memq twisty-paint--current twisty-paint--order)))
         (next (or (cl-find-if (lambda (f) (/= (nth f home) face)) tail)
                   (car twisty-paint--order))))
    (setq twisty-paint--current next)
    (twisty-paint--draw)))

(defun twisty-paint-colour ()
  "Paint the sticker under the cursor with the colour of the key pressed."
  (interactive)
  (let ((index (cl-position (string last-command-event) (twisty-paint--get 'keys) :test #'string=)))
    (unless index (user-error "No colour on key %c" last-command-event))
    (if (twisty-paint--fixed-p twisty-paint--current)
        (message "Centres are fixed; they tell you how to hold the puzzle")
      (aset twisty-paint--colors twisty-paint--current index)
      (setq twisty-paint--message "")
      (twisty-paint--step 1))
    (twisty-paint--draw)))

(defun twisty-paint-reset ()
  "Start again from the solved colouring."
  (interactive)
  (setq twisty-paint--colors (vconcat (twisty-paint--get 'solved))
        twisty-paint--current (car twisty-paint--order)
        twisty-paint--message "")
  (twisty-paint--draw))

(defun twisty-paint-load ()
  "Copy the puzzle currently shown by the daemon into the editor."
  (interactive)
  (setq twisty-paint--colors (vconcat (alist-get 'colors (twisty-request-sync "state")))
        twisty-paint--message "")
  (twisty-paint--draw))

(defun twisty-paint--say (buffer text)
  "Show TEXT as the verdict in paint BUFFER."
  (when (buffer-live-p buffer)
    (with-current-buffer buffer
      (setq twisty-paint--message text)
      (twisty-paint--draw))))

(defun twisty-paint--solved (buffer colors result error)
  "Handle the solve RESULT (or ERROR) for COLORS entered in paint BUFFER."
  (if (or error (not (alist-get 'solved result)))
      (twisty-paint--say buffer (format "No solution found: %s"
                                        (or (alist-get 'message error) (alist-get 'reason result))))
    (twisty-paint--say buffer "")
    (twisty-steps-open result colors)))

(defun twisty-paint-check ()
  "Validate the colouring; solve it when it is a reachable state."
  (interactive)
  (let ((buffer (current-buffer))
        (colors (append twisty-paint--colors nil)))
    (twisty-paint--say buffer "checking …")
    (twisty-request
     "set_colors" `((colors . ,(vconcat colors)))
     (lambda (verdict error)
       (cond
        (error (twisty-paint--say buffer (alist-get 'message error)))
        ((not (alist-get 'accepted verdict))
         (twisty-paint--say buffer (concat "Not possible: " (alist-get 'reason verdict))))
        (t
         (twisty-paint--say buffer "Valid. Solving …")
         (twisty-request "solve" '((method . "auto") (time_limit . 20))
                         (lambda (result err2)
                           (twisty-paint--solved buffer colors result err2)))))))))

(defun twisty-paint-quit ()
  "Leave the paint buffer."
  (interactive)
  (quit-window t))

(defvar twisty-paint-mode-map
  (let ((map (make-sparse-keymap)))
    (suppress-keymap map t)
    (define-key map (kbd "<left>") #'twisty-paint-left)
    (define-key map (kbd "<right>") #'twisty-paint-right)
    (define-key map (kbd "<up>") #'twisty-paint-up)
    (define-key map (kbd "<down>") #'twisty-paint-down)
    (define-key map (kbd "DEL") #'twisty-paint-back)
    (define-key map (kbd "TAB") #'twisty-paint-next-face)
    (define-key map (kbd "RET") #'twisty-paint-check)
    (define-key map (kbd "C-c C-c") #'twisty-paint-check)
    (define-key map (kbd "C-c C-r") #'twisty-paint-reset)
    (define-key map (kbd "C-c C-l") #'twisty-paint-load)
    (define-key map (kbd "C-c C-k") #'twisty-paint-quit)
    map)
  "Keymap of `twisty-paint-mode'; colour keys are added per puzzle.")

(define-derived-mode twisty-paint-mode special-mode "Twisty-Paint"
  "Paint the stickers of a physical puzzle.

\\{twisty-paint-mode-map}"
  (setq-local truncate-lines t)
  (setq-local line-spacing 0)
  (buffer-disable-undo))

(defun twisty-paint--setup (layout)
  "Initialise the current buffer from LAYOUT."
  (twisty-paint-mode)
  (setq twisty-paint--layout layout)
  (setq twisty-paint--colors (vconcat (alist-get 'solved layout))
        twisty-paint--order (twisty-paint--typing-order)
        twisty-paint--current (or (car twisty-paint--order) 0)
        twisty-paint--message "")
  (let ((map (make-sparse-keymap)))
    (set-keymap-parent map twisty-paint-mode-map)
    (dolist (key (alist-get 'keys layout))
      (define-key map (kbd key) #'twisty-paint-colour))
    (use-local-map map))
  (twisty-paint--draw))

;;;###autoload
(defun twisty-paint ()
  "Enter the colours of a physical puzzle and get a solution for it."
  (interactive)
  (twisty-connect)
  (let ((layout (twisty-request-sync "paint_layout"))
        (buffer (get-buffer-create twisty-paint-buffer-name)))
    (pop-to-buffer-same-window buffer)
    (twisty-paint--setup layout)
    buffer))

;;;; Solution stepper ----------------------------------------------------------------------------

(defvar-local twisty-steps--steps nil "Solution as a list of notation strings (U2, R', ...).")
(defvar-local twisty-steps--index 0 "Number of steps already played.")
(defvar-local twisty-steps--result nil)
(defvar-local twisty-steps--colors nil "Colouring the solution starts from.")
(defvar-local twisty-steps--auto nil "Timer of the automatic playback.")

(defun twisty-steps--inverse (step)
  "Notation of the inverse of STEP."
  (if (string-suffix-p "'" step) (substring step 0 -1) (concat step "'")))

(defun twisty-steps--count (step)
  "Number of quarter turns in STEP."
  (if (string-match "\\([0-9]+\\)'?\\'" step) (string-to-number (match-string 1 step)) 1))

(defun twisty-steps--draw ()
  "Redraw the solution buffer."
  (let* ((inhibit-read-only t)
         (total (length twisty-steps--steps))
         (win (get-buffer-window (current-buffer)))
         (width (max 24 (- (if win (window-body-width win) 80) (car twisty-corner-size) 6)))
         (col 2))
    (erase-buffer)
    (insert (propertize "Solution" 'face 'twisty-title) "\n"
            (propertize (format "%d quarter turns in %d steps%s"
                                (length (alist-get 'moves twisty-steps--result)) total
                                (if (alist-get 'optimal_proven twisty-steps--result)
                                    " · optimal" " · not optimal (chain solver)"))
                        'face 'twisty-dim)
            "\n\n")
    (insert (twisty--kv "step" (format "%d / %d" twisty-steps--index total)) "\n"
            (twisty--kv "next" (if (< twisty-steps--index total)
                                   (propertize (nth twisty-steps--index twisty-steps--steps) 'face 'success)
                                 (propertize "solved" 'face 'success)))
            "\n\n  ")
    (cl-loop for step in twisty-steps--steps for i from 0
             do (when (> (+ col (length step) 1) width)
                  (insert "\n  ")
                  (setq col 2))
             (insert (propertize step 'face (cond ((< i twisty-steps--index) 'twisty-dim)
                                                  ((= i twisty-steps--index) '(:inverse-video t :weight bold))
                                                  (t 'default)))
                     " ")
             (setq col (+ col (length step) 1)))
    (insert "\n\n"
            (propertize "  n or SPC next · p back · a play all · . stop · 0 start over · q quit"
                        'face 'twisty-dim))
    (goto-char (point-min))))

(defun twisty-steps-next ()
  "Play the next step of the solution."
  (interactive)
  (if (>= twisty-steps--index (length twisty-steps--steps))
      (progn (twisty-steps-stop) (message "Solved"))
    (twisty-request "play" `((moves . ,(nth twisty-steps--index twisty-steps--steps))))
    (cl-incf twisty-steps--index)
    (twisty-steps--draw)))

(defun twisty-steps-back ()
  "Undo the last played step."
  (interactive)
  (when (> twisty-steps--index 0)
    (cl-decf twisty-steps--index)
    (twisty-request "play" `((moves . ,(twisty-steps--inverse (nth twisty-steps--index twisty-steps--steps)))))
    (twisty-steps--draw)))

(defun twisty-steps-stop ()
  "Stop the automatic playback."
  (interactive)
  (when twisty-steps--auto
    (cancel-timer twisty-steps--auto)
    (setq twisty-steps--auto nil)))

(defun twisty-steps--auto-tick (buffer)
  "Advance BUFFER's playback by one step and schedule the next."
  (when (buffer-live-p buffer)
    (with-current-buffer buffer
      (setq twisty-steps--auto nil)
      (when (< twisty-steps--index (length twisty-steps--steps))
        (let ((turns (twisty-steps--count (nth twisty-steps--index twisty-steps--steps))))
          (twisty-steps-next)
          (setq twisty-steps--auto
                (run-with-timer (+ 0.12 (* turns twisty-move-time)) nil
                                #'twisty-steps--auto-tick buffer)))))))

(defun twisty-steps-auto ()
  "Play the rest of the solution automatically."
  (interactive)
  (twisty-steps-stop)
  (twisty-steps--auto-tick (current-buffer)))

(defun twisty-steps-restart ()
  "Put the puzzle back into the entered state and start over."
  (interactive)
  (twisty-steps-stop)
  (twisty-request-sync "set_colors" `((colors . ,(vconcat twisty-steps--colors))))
  (setq twisty-steps--index 0)
  (twisty-steps--draw))

(defun twisty-steps-quit ()
  "Leave the solution and give the widget back."
  (interactive)
  (twisty-steps-stop)
  (twisty-widget-pop)
  (quit-window t))

(defvar twisty-steps-mode-map
  (let ((map (make-sparse-keymap)))
    (suppress-keymap map t)
    (define-key map (kbd "n") #'twisty-steps-next)
    (define-key map (kbd "SPC") #'twisty-steps-next)
    (define-key map (kbd "<right>") #'twisty-steps-next)
    (define-key map (kbd "p") #'twisty-steps-back)
    (define-key map (kbd "<left>") #'twisty-steps-back)
    (define-key map (kbd "DEL") #'twisty-steps-back)
    (define-key map (kbd "a") #'twisty-steps-auto)
    (define-key map (kbd ".") #'twisty-steps-stop)
    (define-key map (kbd "0") #'twisty-steps-restart)
    (define-key map (kbd "q") #'twisty-steps-quit)
    map)
  "Keymap of `twisty-steps-mode'.")

(define-derived-mode twisty-steps-mode special-mode "Twisty-Solution"
  "Step through a solution while the 3D puzzle follows.

\\{twisty-steps-mode-map}"
  (setq-local truncate-lines t)
  (setq-local line-spacing 0)
  (buffer-disable-undo))

(defun twisty-steps-open (result colors)
  "Show the solution RESULT of the puzzle coloured COLORS."
  (let ((buffer (get-buffer-create twisty-steps-buffer-name)))
    (pop-to-buffer-same-window buffer)
    (twisty-steps-mode)
    (setq twisty-steps--result result
          twisty-steps--steps (alist-get 'notation result)
          twisty-steps--index 0
          twisty-steps--colors colors)
    (twisty-widget-push buffer :show 'spin :caption nil :position 'top-right)
    (twisty-steps--draw)
    buffer))

;;;; Batch self-test -----------------------------------------------------------------------------

(defun twisty-paint-self-test ()
  "Exercise painting, diagnosis and the stepper against a running daemon."
  (let ((twisty-socket-path (or (getenv "TWISTY_SOCKET") twisty-socket-path)))
    (twisty-connect t)
    (twisty-request-sync "set_puzzle" '((name . "cube3")))
    (with-current-buffer (twisty-paint)
      (cl-assert (= (length twisty-paint--colors) 54))
      (cl-assert (= (length twisty-paint--order) 48))
      (cl-assert (string-match-p "Paint your cube3" (buffer-string)))
      (cl-assert (string-match-p "\\[\\]" (buffer-string)))
      ;; typing a colour paints the sticker and advances
      (let ((first twisty-paint--current) (last-command-event ?r))
        (twisty-paint-colour)
        (cl-assert (= (aref twisty-paint--colors first) 4))
        (cl-assert (/= twisty-paint--current first))
        (twisty-paint-back)
        (cl-assert (= twisty-paint--current first)))
      (twisty-paint-right) (twisty-paint-down) (twisty-paint-next-face)
      ;; nine reds on a face plus one too many -> counted and rejected
      (cl-assert (string-match-p "1[0-9]/9\\|8/9" (buffer-string)))
      (twisty-paint-check)
      (let ((deadline (+ (float-time) 20)))
        (while (and (string-match-p "checking" twisty-paint--message) (< (float-time) deadline))
          (accept-process-output nil 0.05)))
      (cl-assert (string-match-p "wrong number of stickers" twisty-paint--message))
      ;; a real scramble is loaded from the daemon
      (twisty-request-sync "reset")
      (twisty-request-sync "apply" '((moves . "R U F' L2 D B")))
      (twisty-paint-load)
      (cl-assert (not (equal twisty-paint--colors (vconcat (twisty-paint--get 'solved)))))
      ;; the untouched scramble is accepted, solved and stepped through
      (twisty-paint-check)
      (let ((deadline (+ (float-time) 40)))
        (while (and (not (get-buffer twisty-steps-buffer-name)) (< (float-time) deadline))
          (accept-process-output nil 0.05))))
    (with-current-buffer twisty-steps-buffer-name
      (cl-assert (derived-mode-p 'twisty-steps-mode))
      (cl-assert (<= 1 (length twisty-steps--steps) 7))
      (cl-assert (equal (twisty-steps--inverse "R2") "R2'"))
      (cl-assert (equal (twisty-steps--inverse "U'") "U"))
      (cl-assert (= (twisty-steps--count "R2'") 2))
      (twisty-steps-next)
      (twisty-steps-back)
      (cl-assert (= twisty-steps--index 0))
      (twisty-request-sync "style" `((quality . 1)))
      (let ((twisty-move-time 0.01))
        (twisty-request-sync "show" '((move_time . 0.01)))
        (dotimes (_ (length twisty-steps--steps)) (twisty-steps-next)))
      (let ((deadline (+ (float-time) 20)) done)
        (while (and (not done) (< (float-time) deadline))
          (setq done (alist-get 'solved (alist-get 'status (twisty-request-sync "dashboard"))))
          (sleep-for 0.02))
        (cl-assert done))
      (cl-assert (string-match-p "solved" (buffer-string)))
      (twisty-steps-restart)
      (cl-assert (not (alist-get 'solved (twisty-request-sync "status"))))
      (twisty-steps-quit))
    (twisty-request-sync "reset")
    (twisty-disconnect)
    (princ "TWISTY-PAINT-OK\n")))

(provide 'twisty-paint)

;;; twisty-paint.el ends here
