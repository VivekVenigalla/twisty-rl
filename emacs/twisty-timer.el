;;; twisty-timer.el --- Speedcubing timer with ao5 / ao12  -*- lexical-binding: t; -*-

;; Package-Requires: ((emacs "28.1"))
;; Keywords: games

;;; Commentary:

;; `twisty-timer' opens a timer buffer.  The 3D puzzle shows a fresh scramble; SPC starts
;; (optionally after a 15 s inspection), any key stops.  Times are stored per puzzle in
;; `twisty-timer-file' together with their scramble, and the buffer shows best, mean, the
;; current and best ao5 / ao12 (WCA trimmed averages) and the latest solves.
;; d marks the last solve DNF, + adds two seconds, DEL deletes it.
;;
;; Without a reachable daemon the timer still works for cubes: scrambles are generated in
;; Emacs and only the 3D picture is missing.

;;; Code:

(require 'cl-lib)
(require 'subr-x)
(require 'twisty-dashboard)

(defcustom twisty-timer-file (locate-user-emacs-file "twisty-times.eld")
  "File the solve times are appended to (one Lisp plist per line)."
  :type 'file :group 'twisty)

(defcustom twisty-timer-inspection nil
  "Non-nil starts every solve with a 15 second inspection countdown.
Starting after 15 s costs +2, after 17 s the solve is a DNF (WCA rules)."
  :type 'boolean :group 'twisty)

(defcustom twisty-timer-scramble-lengths '(("cube2" . 11) ("cube3" . 25) ("cube4" . 40)
                                           ("pyraminx" . 11) ("skewb" . 11) ("megaminx" . 70))
  "Scramble length in quarter turns per puzzle; others use 25."
  :type '(alist :key-type string :value-type integer) :group 'twisty)

(defconst twisty-timer-buffer-name "*twisty-timer*")

(defvar twisty-timer--solves nil "All recorded solves, newest first.")
(defvar twisty-timer--loaded-from nil)
(defvar twisty-timer--state 'idle "One of idle, inspect, running.")
(defvar twisty-timer--t0 0.0)
(defvar twisty-timer--penalty nil "Penalty earned during inspection.")
(defvar twisty-timer--scramble "" "Scramble of the solve in progress.")
(defvar twisty-timer--puzzle "cube3")
(defvar twisty-timer--display 0.0 "Seconds currently shown in big digits.")
(defvar twisty-timer--ticker nil)
(defvar twisty-timer--online nil "Non-nil when the daemon supplied the scramble.")

;;;; Storage -------------------------------------------------------------------------------------

(defun twisty-timer--load ()
  "Read `twisty-timer-file' once per file name."
  (unless (equal twisty-timer--loaded-from twisty-timer-file)
    (setq twisty-timer--solves nil
          twisty-timer--loaded-from twisty-timer-file)
    (when (file-readable-p twisty-timer-file)
      (with-temp-buffer
        (insert-file-contents twisty-timer-file)
        (goto-char (point-min))
        (condition-case nil
            (while t (push (read (current-buffer)) twisty-timer--solves))
          (end-of-file nil)
          (invalid-read-syntax nil))))))

(defun twisty-timer--save-all ()
  "Rewrite the whole file (after editing or deleting a solve)."
  (make-directory (file-name-directory (expand-file-name twisty-timer-file)) t)
  (with-temp-file twisty-timer-file
    (dolist (solve (reverse twisty-timer--solves))
      (prin1 solve (current-buffer))
      (insert "\n"))))

(defun twisty-timer--record (seconds penalty)
  "Store a solve of SECONDS with PENALTY for the current puzzle."
  (twisty-timer--load)
  (let ((solve (list :time (/ (fround (* seconds 1000)) 1000.0) :penalty penalty
                     :puzzle twisty-timer--puzzle :scramble twisty-timer--scramble
                     :date (format-time-string "%FT%T"))))
    (push solve twisty-timer--solves)
    (make-directory (file-name-directory (expand-file-name twisty-timer-file)) t)
    (let ((print-length nil) (print-level nil))
      (write-region (concat (prin1-to-string solve) "\n") nil twisty-timer-file 'append 'silent))
    solve))

;;;; Statistics ----------------------------------------------------------------------------------

(defun twisty-timer--value (solve)
  "Effective seconds of SOLVE: +2 applied, a DNF is infinite."
  (pcase (plist-get solve :penalty)
    ('dnf 1.0e+INF)
    ('plus2 (+ 2.0 (plist-get solve :time)))
    (_ (plist-get solve :time))))

(defun twisty-timer--average (values)
  "WCA trimmed average of VALUES: best and worst dropped, two DNFs spoil it."
  (let ((sorted (sort (copy-sequence values) #'<)))
    (if (>= (cl-count 1.0e+INF sorted) 2)
        1.0e+INF
      (let ((middle (butlast (cdr sorted))))
        (/ (apply #'+ middle) (float (length middle)))))))

(defun twisty-timer-stats (&optional puzzle)
  "Statistics plist for PUZZLE (default: the timer's current puzzle)."
  (twisty-timer--load)
  (let* ((puzzle (or puzzle twisty-timer--puzzle))
         (values (mapcar #'twisty-timer--value
                         (cl-remove-if-not (lambda (s) (equal (plist-get s :puzzle) puzzle))
                                           twisty-timer--solves)))   ; newest first
         (finite (cl-remove 1.0e+INF values))
         (rolling (lambda (n)
                    (cl-loop for tail on values while (>= (length tail) n)
                             minimize (twisty-timer--average (seq-take tail n))))))
    (list :count (length values)
          :best (and finite (apply #'min finite))
          :mean (and finite (/ (apply #'+ finite) (float (length finite))))
          :ao5 (and (>= (length values) 5) (twisty-timer--average (seq-take values 5)))
          :ao12 (and (>= (length values) 12) (twisty-timer--average (seq-take values 12)))
          :best-ao5 (and (>= (length values) 5) (funcall rolling 5))
          :best-ao12 (and (>= (length values) 12) (funcall rolling 12))
          :recent (seq-take (cl-remove-if-not (lambda (s) (equal (plist-get s :puzzle) puzzle))
                                              twisty-timer--solves)
                            12))))

(defun twisty-timer-format (seconds)
  "Format SECONDS like a stackmat display; nil is a dash and infinity is DNF."
  (cond ((null seconds) "—")
        ((= seconds 1.0e+INF) "DNF")
        ((>= seconds 60) (format "%d:%05.2f" (floor seconds 60) (mod seconds 60.0)))
        (t (format "%.2f" seconds))))

(defun twisty-timer--format-solve (solve)
  "Format SOLVE with its penalty mark."
  (pcase (plist-get solve :penalty)
    ('dnf "DNF")
    ('plus2 (concat (twisty-timer-format (+ 2.0 (plist-get solve :time))) "+"))
    (_ (twisty-timer-format (plist-get solve :time)))))

(defun twisty-timer-summary (&optional puzzle)
  "One-line summary for PUZZLE, or nil when nothing was recorded yet."
  (let ((stats (twisty-timer-stats (or puzzle twisty-default-puzzle))))
    (when (> (plist-get stats :count) 0)
      (format "%d solves · best %s · ao5 %s · ao12 %s"
              (plist-get stats :count)
              (twisty-timer-format (plist-get stats :best))
              (twisty-timer-format (plist-get stats :ao5))
              (twisty-timer-format (plist-get stats :ao12))))))

;;;; Display -------------------------------------------------------------------------------------

(defconst twisty-timer--font
  '((?0 "###" "# #" "# #" "# #" "###") (?1 " # " "## " " # " " # " "###")
    (?2 "###" "  #" "###" "#  " "###") (?3 "###" "  #" "###" "  #" "###")
    (?4 "# #" "# #" "###" "  #" "  #") (?5 "###" "#  " "###" "  #" "###")
    (?6 "###" "#  " "###" "# #" "###") (?7 "###" "  #" "  #" "  #" "  #")
    (?8 "###" "# #" "###" "# #" "###") (?9 "###" "# #" "###" "  #" "###")
    (?: " " "#" " " "#" " ") (?. " " " " " " " " "#")
    (?D "## " "# #" "# #" "# #" "## ") (?N "# #" "###" "###" "# #" "# #")
    (?F "###" "#  " "## " "#  " "#  ") (?+ "   " " # " "###" " # " "   ")
    (?— "   " "   " "###" "   " "   "))
  "Five-row block font; every lit pixel is drawn two columns wide.")

(defun twisty-timer--big (text face)
  "Render TEXT as five lines of big digits in FACE's foreground colour."
  (let ((colour (or (face-foreground face nil t) "#ffffff")))
    (cl-loop for row from 0 below 5
             collect (mapconcat
                      (lambda (ch)
                        (let ((glyph (or (alist-get ch twisty-timer--font) '("   " "   " "   " "   " "   "))))
                          (concat (mapconcat (lambda (px)
                                               (if (eq px ?#)
                                                   (propertize "  " 'face (list :background colour))
                                                 "  "))
                                             (nth row glyph) "")
                                  "  ")))
                      text ""))))

(defun twisty-timer--draw ()
  "Redraw the timer buffer."
  (let ((buffer (get-buffer twisty-timer-buffer-name)))
    (when (buffer-live-p buffer)
      (with-current-buffer buffer
        (let* ((inhibit-read-only t)
               (stats (twisty-timer-stats))
               (face (pcase twisty-timer--state ('running 'success) ('inspect 'warning) (_ 'default)))
               (win (get-buffer-window buffer))
               (width (max 30 (- (if win (window-body-width win) 80)
                                 (if twisty-timer--online (+ 4 (car twisty-corner-size)) 2)))))
          (erase-buffer)
          (insert (propertize (format "Timer · %s" twisty-timer--puzzle) 'face 'twisty-title) "\n\n")
          (insert (propertize "scramble" 'face 'twisty-key) "\n")
          (let ((fill-column (- width 2)) (fill-prefix "  ") (start (point)))
            (insert "  " twisty-timer--scramble "\n")
            (fill-region start (point)))
          (insert "\n")
          (dolist (line (twisty-timer--big
                         (pcase twisty-timer--state
                           ('inspect (format "%d" (max 0 (ceiling (- 15 twisty-timer--display)))))
                           (_ (twisty-timer-format twisty-timer--display)))
                         face))
            (insert "  " line "\n"))
          (insert "\n"
                  (propertize (pcase twisty-timer--state
                                ('idle "  SPC to start")
                                ('inspect "  inspecting · SPC starts the solve")
                                ('running "  any key stops"))
                              'face 'twisty-dim)
                  "\n\n")
          (insert (format "  %s %-6d %s %-9s %s %s\n"
                          (propertize "solves" 'face 'twisty-key) (plist-get stats :count)
                          (propertize "best" 'face 'twisty-key) (twisty-timer-format (plist-get stats :best))
                          (propertize "mean" 'face 'twisty-key) (twisty-timer-format (plist-get stats :mean)))
                  (format "  %s %-9s %s %-9s\n"
                          (propertize "ao5 " 'face 'twisty-key) (twisty-timer-format (plist-get stats :ao5))
                          (propertize "ao12" 'face 'twisty-key) (twisty-timer-format (plist-get stats :ao12)))
                  (format "  %s %-9s %s %-9s\n\n"
                          (propertize "best ao5 " 'face 'twisty-key)
                          (twisty-timer-format (plist-get stats :best-ao5))
                          (propertize "best ao12" 'face 'twisty-key)
                          (twisty-timer-format (plist-get stats :best-ao12))))
          (insert (propertize "  last  " 'face 'twisty-key)
                  (if (plist-get stats :recent)
                      (truncate-string-to-width
                       (mapconcat #'twisty-timer--format-solve (plist-get stats :recent) "  ")
                       (- width 10) nil nil "…")
                    "—")
                  "\n\n"
                  (propertize (format "  i inspection (%s) · d DNF · + plus two · DEL delete last"
                                      (if twisty-timer-inspection "on" "off"))
                              'face 'twisty-dim)
                  "\n"
                  (propertize "  n new scramble · p puzzle · q quit" 'face 'twisty-dim))
          (goto-char (point-min)))))))

;;;; Scrambles -----------------------------------------------------------------------------------

(defun twisty-timer--local-scramble (length)
  "A random-move cube scramble of about LENGTH quarter turns, generated in Emacs."
  (let ((faces '("U" "D" "L" "R" "F" "B")) (out nil) (last nil) (turns 0))
    (while (< turns length)
      (let ((face (nth (random 6) faces)) (suffix (nth (random 3) '("" "'" "2"))))
        (unless (equal face last)
          (push (concat face suffix) out)
          (setq last face turns (+ turns (if (equal suffix "2") 2 1))))))
    (string-join (nreverse out) " ")))

(defun twisty-timer-new-scramble ()
  "Fetch a fresh scramble and show it on the puzzle."
  (interactive)
  (when (eq twisty-timer--state 'idle)
    (let ((length (alist-get twisty-timer--puzzle twisty-timer-scramble-lengths 25 nil #'equal)))
      (setq twisty-timer--online nil)
      (condition-case nil
          (when (or (twisty--live-p) (twisty-available-p))
            (let ((status (twisty-request-sync "status" nil 20)))
              (unless (equal (alist-get 'puzzle status) twisty-timer--puzzle)
                (twisty-request-sync "set_puzzle" `((name . ,twisty-timer--puzzle)))))
            (setq twisty-timer--scramble
                  (alist-get 'text (twisty-request-sync "scramble" `((depth . ,length))))
                  twisty-timer--online t))
        (error nil))
      (unless twisty-timer--online
        (setq twisty-timer--scramble (twisty-timer--local-scramble length))))
    (twisty-timer--draw)))

;;;; State machine -------------------------------------------------------------------------------

(defun twisty-timer--tick ()
  "Refresh the running clock."
  (if (memq twisty-timer--state '(inspect running))
      (progn
        (setq twisty-timer--display (- (float-time) twisty-timer--t0))
        (when (and (eq twisty-timer--state 'inspect) (>= twisty-timer--display 17.0))
          (setq twisty-timer--penalty 'dnf)
          (twisty-timer--begin))
        (twisty-timer--draw))
    (twisty-timer--stop-ticker)))

(defun twisty-timer--stop-ticker ()
  "Cancel the display timer."
  (when twisty-timer--ticker
    (cancel-timer twisty-timer--ticker)
    (setq twisty-timer--ticker nil)))

(defun twisty-timer--start-ticker ()
  "Start the display timer."
  (twisty-timer--stop-ticker)
  (setq twisty-timer--ticker (run-with-timer 0 0.07 #'twisty-timer--tick)))

(defun twisty-timer--begin ()
  "Start the clock."
  (when (eq twisty-timer--state 'inspect)
    (let ((used (- (float-time) twisty-timer--t0)))
      (setq twisty-timer--penalty (cond ((>= used 17.0) 'dnf) ((>= used 15.0) 'plus2)))))
  (setq twisty-timer--state 'running
        twisty-timer--t0 (float-time)
        twisty-timer--display 0.0)
  (twisty-timer--start-ticker)
  ;; while the clock runs every key (and click) stops it
  (set-transient-map (let ((map (make-sparse-keymap)))
                       (define-key map [t] #'twisty-timer-stop)
                       map)
                     (lambda () (eq twisty-timer--state 'running))))

(defun twisty-timer-stop ()
  "Stop the clock and record the solve."
  (interactive)
  (when (eq twisty-timer--state 'running)
    (let ((elapsed (- (float-time) twisty-timer--t0)))
      (setq twisty-timer--state 'idle
            twisty-timer--display elapsed)
      (twisty-timer--stop-ticker)
      (twisty-timer--record elapsed twisty-timer--penalty)
      (setq twisty-timer--penalty nil)
      (twisty-timer-new-scramble)
      (message "%s" (twisty-timer--format-solve (car twisty-timer--solves))))))

(defun twisty-timer-space ()
  "Start inspection, start the solve or stop it, depending on the state."
  (interactive)
  (pcase twisty-timer--state
    ('idle (setq twisty-timer--penalty nil)
           (if twisty-timer-inspection
               (progn (setq twisty-timer--state 'inspect
                            twisty-timer--t0 (float-time)
                            twisty-timer--display 0.0)
                      (twisty-timer--start-ticker))
             (twisty-timer--begin))
           (twisty-timer--draw))
    ('inspect (twisty-timer--begin) (twisty-timer--draw))
    ('running (twisty-timer-stop))))

(defun twisty-timer--edit-last (penalty)
  "Toggle PENALTY on the most recent solve of the current puzzle."
  (twisty-timer--load)
  (let ((solve (cl-find-if (lambda (s) (equal (plist-get s :puzzle) twisty-timer--puzzle))
                           twisty-timer--solves)))
    (unless solve (user-error "No solve to edit"))
    (plist-put solve :penalty (unless (eq (plist-get solve :penalty) penalty) penalty))
    (twisty-timer--save-all)
    (twisty-timer--draw)))

(defun twisty-timer-dnf () "Toggle DNF on the last solve." (interactive) (twisty-timer--edit-last 'dnf))
(defun twisty-timer-plus2 () "Toggle +2 on the last solve." (interactive) (twisty-timer--edit-last 'plus2))

(defun twisty-timer-delete-last ()
  "Delete the most recent solve of the current puzzle."
  (interactive)
  (twisty-timer--load)
  (let ((solve (cl-find-if (lambda (s) (equal (plist-get s :puzzle) twisty-timer--puzzle))
                           twisty-timer--solves)))
    (unless solve (user-error "No solve to delete"))
    (when (or noninteractive
              (y-or-n-p (format "Delete %s? " (twisty-timer--format-solve solve))))
      (setq twisty-timer--solves (delq solve twisty-timer--solves))
      (twisty-timer--save-all)
      (twisty-timer--draw))))

(defun twisty-timer-toggle-inspection ()
  "Switch the inspection countdown on or off."
  (interactive)
  (setq twisty-timer-inspection (not twisty-timer-inspection))
  (twisty-timer--draw))

(defun twisty-timer-select-puzzle (name)
  "Time puzzle NAME from now on."
  (interactive
   (list (completing-read "Puzzle: " (condition-case nil (twisty-request-sync "list_puzzles")
                                       (error '("cube2" "cube3" "cube4")))
                          nil t)))
  (setq twisty-timer--puzzle name
        twisty-timer--display 0.0)
  (twisty-timer-new-scramble))

(defun twisty-timer-busy-p ()
  "Non-nil while a solve or its inspection is being timed."
  (memq twisty-timer--state '(inspect running)))

(defun twisty-timer-quit ()
  "Leave the timer and give the widget back."
  (interactive)
  (twisty-timer--stop-ticker)
  (setq twisty-timer--state 'idle)
  (when twisty-timer--online
    (twisty-widget-pop))
  (quit-window t))

(defvar twisty-timer-mode-map
  (let ((map (make-sparse-keymap)))
    (suppress-keymap map t)
    (define-key map (kbd "SPC") #'twisty-timer-space)
    (define-key map (kbd "RET") #'twisty-timer-space)
    (define-key map (kbd "d") #'twisty-timer-dnf)
    (define-key map (kbd "+") #'twisty-timer-plus2)
    (define-key map (kbd "=") #'twisty-timer-plus2)
    (define-key map (kbd "DEL") #'twisty-timer-delete-last)
    (define-key map (kbd "i") #'twisty-timer-toggle-inspection)
    (define-key map (kbd "n") #'twisty-timer-new-scramble)
    (define-key map (kbd "p") #'twisty-timer-select-puzzle)
    (define-key map (kbd "q") #'twisty-timer-quit)
    map)
  "Keymap of `twisty-timer-mode'.")

(define-derived-mode twisty-timer-mode special-mode "Twisty-Timer"
  "Speedcubing timer.

\\{twisty-timer-mode-map}"
  (setq-local truncate-lines t)
  (setq-local line-spacing 0)
  (setq-local cursor-type nil)
  (buffer-disable-undo))

;;;###autoload
(defun twisty-timer (&optional puzzle)
  "Open the speedcubing timer for PUZZLE (default: the daemon's current puzzle)."
  (interactive)
  (let ((buffer (get-buffer-create twisty-timer-buffer-name)))
    (pop-to-buffer-same-window buffer)
    (unless (derived-mode-p 'twisty-timer-mode)
      (twisty-timer-mode))
    (setq twisty-timer--state 'idle
          twisty-timer--display 0.0
          twisty-timer--puzzle (or puzzle
                                   (alist-get 'puzzle twisty--status)
                                   twisty-default-puzzle))
    (twisty-timer--load)
    (twisty-timer-new-scramble)
    (when twisty-timer--online
      (twisty-widget-push buffer :show 'spin :caption nil :position 'top-right))
    (twisty-timer--draw)
    buffer))

;;;; Batch self-test -----------------------------------------------------------------------------

(defun twisty-timer-self-test ()
  "Check statistics, storage and the state machine (daemon optional)."
  (let* ((twisty-timer-file (make-temp-file "twisty-times" nil ".eld"))
         (twisty-socket-path (or (getenv "TWISTY_SOCKET") twisty-socket-path))
         (twisty-timer--solves nil) (twisty-timer--loaded-from nil)
         (twisty-timer--puzzle "cube3") (twisty-timer--scramble "R U2 F'"))
    ;; WCA averages
    (cl-assert (= (twisty-timer--average '(1.0 2.0 3.0 4.0 100.0)) 3.0))
    (cl-assert (= (twisty-timer--average (list 1.0 2.0 3.0 4.0 1.0e+INF)) 3.0))
    (cl-assert (= (twisty-timer--average (list 1.0 1.0e+INF 3.0 4.0 1.0e+INF)) 1.0e+INF))
    (cl-assert (equal (twisty-timer-format 9.814) "9.81"))
    (cl-assert (equal (twisty-timer-format 75.5) "1:15.50"))
    (cl-assert (equal (twisty-timer-format 1.0e+INF) "DNF"))
    (cl-assert (equal (twisty-timer-format nil) "—"))
    (cl-assert (null (twisty-timer-summary "cube3")))
    ;; storage and statistics
    (dolist (seconds '(12.0 10.0 14.0 11.0 13.0 20.0 9.0 15.0 16.0 12.5 13.5 11.5))
      (twisty-timer--record seconds nil))
    (let ((stats (twisty-timer-stats)))
      (cl-assert (= (plist-get stats :count) 12))
      (cl-assert (= (plist-get stats :best) 9.0))
      (cl-assert (< (abs (- (plist-get stats :ao5) (/ (+ 12.5 13.5 15.0) 3.0))) 1e-9))
      (cl-assert (< (abs (- (plist-get stats :ao12) 12.85)) 1e-9))
      (cl-assert (<= (plist-get stats :best-ao5) (plist-get stats :ao5))))
    (twisty-timer--edit-last 'plus2)
    (cl-assert (equal (twisty-timer--format-solve (car twisty-timer--solves)) "13.50+"))
    (twisty-timer--edit-last 'dnf)
    (cl-assert (equal (twisty-timer--format-solve (car twisty-timer--solves)) "DNF"))
    (twisty-timer-delete-last)
    (cl-assert (= (plist-get (twisty-timer-stats) :count) 11))
    (twisty-timer--record 30.0 nil)
    (let ((twisty-timer--puzzle "pyraminx")) (twisty-timer--record 5.0 nil))
    ;; a fresh load sees exactly what was written
    (setq twisty-timer--solves nil twisty-timer--loaded-from nil)
    (cl-assert (= (plist-get (twisty-timer-stats "cube3") :count) 12))
    (cl-assert (= (plist-get (twisty-timer-stats "pyraminx") :best) 5.0))
    (cl-assert (string-match-p "12 solves · best 9.00" (twisty-timer-summary "cube3")))
    ;; scrambles and big digits
    (cl-assert (string-match-p "\\`\\([UDLRFB]['2]? ?\\)+\\'" (twisty-timer--local-scramble 25)))
    (cl-assert (= (length (twisty-timer--big "12.34" 'default)) 5))
    ;; the state machine, against the daemon when one is reachable
    (when (getenv "TWISTY_SOCKET") (twisty-connect t))
    (with-current-buffer (twisty-timer "cube3")
      (cl-assert (eq twisty-timer--state 'idle))
      (cl-assert (> (length twisty-timer--scramble) 20))
      (cl-assert (eq twisty-timer--online (and (getenv "TWISTY_SOCKET") t)))
      (let ((first twisty-timer--scramble) (twisty-timer-inspection t))
        (twisty-timer-space)
        (cl-assert (eq twisty-timer--state 'inspect))
        (cl-assert (twisty-timer-busy-p))
        (twisty-timer-space)
        (cl-assert (eq twisty-timer--state 'running))
        (sleep-for 0.25)
        (twisty-timer--tick)
        (cl-assert (string-match-p "any key stops" (buffer-string)))
        (twisty-timer-stop)
        (cl-assert (eq twisty-timer--state 'idle))
        (cl-assert (not (twisty-timer-busy-p)))
        (cl-assert (not (equal first twisty-timer--scramble))))
      (let ((solve (car twisty-timer--solves)))
        (cl-assert (< 0.2 (plist-get solve :time) 2.0))
        (cl-assert (null (plist-get solve :penalty))))
      (cl-assert (= (plist-get (twisty-timer-stats "cube3") :count) 13))
      (cl-assert (string-match-p "best ao12" (buffer-string)))
      (twisty-timer-quit))
    (delete-file twisty-timer-file)
    (when (twisty--live-p) (twisty-disconnect))
    (princ "TWISTY-TIMER-OK\n")))

(provide 'twisty-timer)

;;; twisty-timer.el ends here
