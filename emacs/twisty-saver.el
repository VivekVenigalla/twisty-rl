;;; twisty-saver.el --- Idle screensaver: the puzzle takes over the window  -*- lexical-binding: t; -*-

;; Package-Requires: ((emacs "28.1"))
;; Keywords: games

;;; Commentary:

;; With `twisty-saver-mode' on, the puzzle fills the selected frame after
;; `twisty-saver-idle-minutes' without input and keeps scrambling and solving itself, rotating
;; through `twisty-saver-puzzles'.  The next key press or click restores the windows exactly as
;; they were (the key is swallowed).  `twisty-saver-start' shows it immediately.

;;; Code:

(require 'cl-lib)
(require 'twisty-dashboard)

(defcustom twisty-saver-idle-minutes 5
  "Idle minutes before the screensaver starts."
  :type 'number :group 'twisty)

(defcustom twisty-saver-puzzles '("cube3" "megaminx" "pyraminx" "skewb" "cube2")
  "Puzzles the screensaver rotates through; nil keeps the current one."
  :type '(repeat string) :group 'twisty)

(defcustom twisty-saver-max-size '(120 . 60)
  "Largest (COLUMNS . ROWS) the screensaver puzzle may take."
  :type '(cons integer integer) :group 'twisty)

(defvar twisty-saver-inhibit-functions nil
  "Functions called without arguments; a non-nil result vetoes the screensaver.")

(defconst twisty-saver-buffer-name "*twisty-saver*")
(defvar twisty-saver--active nil)
(defvar twisty-saver--windows nil "Window configuration to restore.")
(defvar twisty-saver--idle-timer nil)

(defun twisty-saver--size (window)
  "Puzzle size and top margin for WINDOW, as ((COLUMNS . ROWS) . MARGIN)."
  (let* ((width (window-body-width window))
         (height (window-body-height window))
         (rows (max 8 (min (cdr twisty-saver-max-size) (- height 4) (/ (* (- width 2) 4) 7))))
         (cols (min (car twisty-saver-max-size) (/ (* rows 7) 4))))
    (cons (cons cols rows) (max 0 (/ (- height rows 3) 2)))))

;;;###autoload
(defun twisty-saver-start ()
  "Let the puzzle take over the frame until the next key press."
  (interactive)
  (unless twisty-saver--active
    (setq twisty-saver--active t
          twisty-saver--windows (current-window-configuration))
    (let ((buffer (get-buffer-create twisty-saver-buffer-name)))
      (delete-other-windows)
      (switch-to-buffer buffer t t)
      (with-current-buffer buffer
        (special-mode)
        (setq-local cursor-type nil mode-line-format nil truncate-lines t line-spacing 0)
        (let ((inhibit-read-only t)) (erase-buffer)))
      (let ((layout (twisty-saver--size (selected-window))))
        (twisty-widget-push buffer :size (car layout) :margin (cdr layout)
                            :position 'top-center :caption t :show 'attract
                            :carousel twisty-saver-puzzles)))
    ;; swallow the key (or click) that wakes Emacs up
    (set-transient-map (let ((map (make-sparse-keymap)))
                         (define-key map [t] #'twisty-saver-stop)
                         map)
                       (lambda () twisty-saver--active))))

(defun twisty-saver-stop ()
  "Leave the screensaver and restore the windows."
  (interactive)
  (when twisty-saver--active
    (setq twisty-saver--active nil)
    (twisty-widget-pop)
    (when (window-configuration-p twisty-saver--windows)
      (set-window-configuration twisty-saver--windows))
    (setq twisty-saver--windows nil)
    (when (get-buffer twisty-saver-buffer-name)
      (kill-buffer twisty-saver-buffer-name))))

(defun twisty-saver--maybe-start ()
  "Start the screensaver from the idle timer unless something vetoes it."
  (when (and (not twisty-saver--active)
             (not noninteractive)
             (not (active-minibuffer-window))
             (not executing-kbd-macro)
             (frame-visible-p (selected-frame))
             (twisty-available-p)
             (not (run-hook-with-args-until-success 'twisty-saver-inhibit-functions)))
    (twisty-saver-start)))

;;;###autoload
(define-minor-mode twisty-saver-mode
  "Start the puzzle screensaver after `twisty-saver-idle-minutes' of idleness."
  :global t :group 'twisty
  (when twisty-saver--idle-timer
    (cancel-timer twisty-saver--idle-timer)
    (setq twisty-saver--idle-timer nil))
  (when (and twisty-saver-mode twisty-saver-idle-minutes (> twisty-saver-idle-minutes 0))
    (setq twisty-saver--idle-timer
          (run-with-idle-timer (* 60 twisty-saver-idle-minutes) t #'twisty-saver--maybe-start))))

;;;; Batch self-test -----------------------------------------------------------------------------

(defun twisty-saver-self-test ()
  "Check start / stop bookkeeping against a running daemon."
  (let ((twisty-socket-path (or (getenv "TWISTY_SOCKET") twisty-socket-path))
        (twisty-saver-puzzles '("cube2" "pyraminx"))
        (twisty-show 'spin) (twisty-corner-size '(30 . 15)) (twisty-widget-position 'top-right))
    (twisty-connect t)
    (twisty-request-sync "set_puzzle" '((name . "cube3")))
    (setq twisty--status (twisty-request-sync "status"))
    (with-temp-buffer
      (insert "work in progress\n")
      (switch-to-buffer (current-buffer))
      (let ((before (current-buffer)))
        (setq twisty--corner-buffer before)
        (twisty-saver-start)
        (cl-assert twisty-saver--active)
        (cl-assert (eq (window-buffer) (get-buffer twisty-saver-buffer-name)))
        (cl-assert (eq twisty--corner-buffer (get-buffer twisty-saver-buffer-name)))
        (cl-assert (eq twisty-show 'attract))
        (cl-assert (eq twisty-widget-position 'top-center))
        (cl-assert (not (equal twisty-corner-size '(30 . 15))))
        (accept-process-output nil 0.3)                   ; the show request is served by a worker thread
        (let ((show (alist-get 'show (twisty-request-sync "status"))))
          (cl-assert (equal (alist-get 'mode show) "attract"))
          (cl-assert (equal (alist-get 'carousel show) '("cube2" "pyraminx"))))
        (twisty-saver-start)                              ; idempotent
        (cl-assert (= (length twisty--widget-stack) 1))
        (twisty-saver-stop)
        (cl-assert (not twisty-saver--active))
        (cl-assert (eq (window-buffer) before))
        (cl-assert (eq twisty--corner-buffer before))
        (cl-assert (eq twisty-show 'spin))
        (cl-assert (equal twisty-corner-size '(30 . 15)))
        (cl-assert (null twisty--widget-stack))
        (cl-assert (not (get-buffer twisty-saver-buffer-name)))
        (accept-process-output nil 0.3)
        (let ((status (twisty-request-sync "status")))
          (cl-assert (equal (alist-get 'puzzle status) "cube3"))
          (cl-assert (equal (alist-get 'mode (alist-get 'show status)) "spin"))
          (cl-assert (null (alist-get 'carousel (alist-get 'show status)))))
        ;; vetoes
        (let ((twisty-saver-inhibit-functions (list (lambda () t))))
          (cl-assert (run-hook-with-args-until-success 'twisty-saver-inhibit-functions))
          (twisty-saver--maybe-start)
          (cl-assert (not twisty-saver--active)))))
    (twisty-disconnect)
    (princ "TWISTY-SAVER-OK\n")))

(provide 'twisty-saver)

;;; twisty-saver.el ends here
