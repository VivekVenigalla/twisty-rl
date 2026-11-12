;;; twisty-dashboard.el --- TwistyRL 3D puzzle widget and interactive solver  -*- lexical-binding: t; -*-

;; Author: TwistyRL
;; Version: 0.1.0
;; Package-Requires: ((emacs "28.1"))
;; Keywords: games, tools

;;; Commentary:

;; Client for the TwistyRL daemon (`python main.py daemon'), speaking newline-delimited
;; JSON-RPC 2.0 over a Unix domain socket.
;;
;;   (add-to-list 'load-path "/path/to/twisty-rl/emacs")
;;   (require 'twisty-dashboard)
;;   (twisty-dashboard-mode 1)      ; rotating 3D puzzle in the corner of the startup buffer
;;   M-x twisty-dashboard           ; full interactive dashboard
;;
;; `twisty-dashboard-mode' is a global minor mode: it starts the daemon when needed and paints a
;; rotating Unicode half-block rendering of the puzzle into the top-right corner of the startup
;; buffer (or any buffer, see `twisty-dashboard-attach').  `twisty-dashboard' opens a dedicated
;; buffer in `twisty-mode' where the puzzle can be turned, scrambled, solved by the search engine
;; (the solution is played back move by move), analysed with Schreier-Sims, mined for commutator
;; macros and used to train the network while the Bellman loss is plotted live.

;;; Code:

(require 'cl-lib)
(require 'json)
(require 'subr-x)

(defgroup twisty nil
  "TwistyRL puzzle dashboard."
  :group 'games
  :prefix "twisty-")

(defcustom twisty-project-directory
  (let ((here (or load-file-name buffer-file-name default-directory)))
    (file-name-directory (directory-file-name (file-name-directory here))))
  "Root of the twisty-rl checkout (the directory containing main.py)."
  :type 'directory)

(defcustom twisty-python-command
  (let ((venv (expand-file-name ".venv/bin/python" twisty-project-directory)))
    (if (file-executable-p venv) venv "python3"))
  "Python interpreter used to launch the TwistyRL daemon.
Defaults to the project's virtualenv when it exists."
  :type 'string)

(defcustom twisty-socket-path
  (expand-file-name (format "twisty-rl-%d.sock" (user-uid))
                    (or (getenv "XDG_RUNTIME_DIR") temporary-file-directory))
  "Unix domain socket of the TwistyRL daemon."
  :type 'file)

(defcustom twisty-default-puzzle "cube3"
  "Puzzle loaded when the daemon is started from Emacs."
  :type 'string)

(defcustom twisty-fps 10
  "Refresh rate of the rotating puzzle, in frames per second."
  :type 'number)

(defcustom twisty-corner-size '(30 . 15)
  "Size (COLUMNS . ROWS) of the corner widget."
  :type '(cons integer integer))

(defcustom twisty-frame-size '(46 . 23)
  "Size (COLUMNS . ROWS) of the 3D view in the dashboard buffer."
  :type '(cons integer integer))

(defcustom twisty-request-timeout 60
  "Seconds to wait for a synchronous daemon reply."
  :type 'number)

(defcustom twisty-playback-delay 0.18
  "Seconds between moves when a solution is played back."
  :type 'number)

(defface twisty-title '((t :inherit font-lock-keyword-face :weight bold :height 1.1))
  "Face of dashboard headings.")
(defface twisty-key '((t :inherit font-lock-constant-face :weight bold))
  "Face of key hints.")
(defface twisty-dim '((t :inherit shadow))
  "Face of secondary dashboard text.")

;;;; Connection ----------------------------------------------------------------------------------

(defvar twisty--daemon-process nil "Daemon process started by Emacs, if any.")
(defvar twisty--connection nil "Network process connected to the daemon.")
(defvar twisty--next-id 0)
(defvar twisty--callbacks (make-hash-table :test 'eql))
(defvar twisty--partial "" "Unterminated tail of the daemon's output.")

(defun twisty--live-p ()
  "Return non-nil when the RPC connection is open."
  (and twisty--connection (process-live-p twisty--connection)))

(defun twisty--filter (_proc chunk)
  "Accumulate CHUNK and dispatch every complete JSON-RPC line."
  (setq twisty--partial (concat twisty--partial chunk))
  (let (pos)
    (while (setq pos (string-search "\n" twisty--partial))
      (let ((line (substring twisty--partial 0 pos)))
        (setq twisty--partial (substring twisty--partial (1+ pos)))
        (unless (string-empty-p line)
          (condition-case err
              (let* ((msg (json-parse-string line :object-type 'alist :array-type 'list
                                             :null-object nil :false-object nil))
                     (id (alist-get 'id msg))
                     (callback (and id (gethash id twisty--callbacks))))
                (when callback
                  (remhash id twisty--callbacks)
                  (funcall callback (alist-get 'result msg) (alist-get 'error msg))))
            (error (message "twisty: bad reply (%s)" (error-message-string err)))))))))

(defun twisty--sentinel (proc _event)
  "Forget PROC and its pending callbacks when the connection closes."
  (unless (process-live-p proc)
    (when (eq proc twisty--connection)
      (setq twisty--connection nil twisty--partial "")
      (clrhash twisty--callbacks))))

(defun twisty-start-daemon ()
  "Launch the TwistyRL daemon as a subprocess and wait for its socket."
  (interactive)
  (unless (and twisty--daemon-process (process-live-p twisty--daemon-process))
    (let ((default-directory (file-name-as-directory twisty-project-directory))
          (process-connection-type nil))
      (setq twisty--daemon-process
            (start-process "twisty-daemon" (get-buffer-create " *twisty-daemon*")
                           twisty-python-command "main.py" "daemon"
                           "--socket" twisty-socket-path "--puzzle" twisty-default-puzzle))
      (set-process-query-on-exit-flag twisty--daemon-process nil)))
  (let ((deadline (+ (float-time) 45)))
    (while (and (not (file-exists-p twisty-socket-path))
                (process-live-p twisty--daemon-process)
                (< (float-time) deadline))
      (accept-process-output twisty--daemon-process 0.1)))
  (unless (file-exists-p twisty-socket-path)
    (error "twisty: daemon did not start (see buffer  *twisty-daemon*)")))

(defun twisty-connect (&optional no-start)
  "Connect to the daemon, starting it first unless NO-START is non-nil."
  (interactive)
  (unless (twisty--live-p)
    (unless (or (file-exists-p twisty-socket-path) no-start)
      (twisty-start-daemon))
    (setq twisty--partial "")
    (clrhash twisty--callbacks)
    (setq twisty--connection
          (condition-case nil
              (make-network-process :name "twisty-rpc" :family 'local :service twisty-socket-path
                                    :coding 'utf-8 :noquery t
                                    :filter #'twisty--filter :sentinel #'twisty--sentinel)
            (file-error
             ;; stale socket file left behind by a dead daemon
             (if no-start
                 (error "twisty: no daemon at %s" twisty-socket-path)
               (ignore-errors (delete-file twisty-socket-path))
               (twisty-start-daemon)
               (make-network-process :name "twisty-rpc" :family 'local :service twisty-socket-path
                                     :coding 'utf-8 :noquery t
                                     :filter #'twisty--filter :sentinel #'twisty--sentinel))))))
  twisty--connection)

(defun twisty-disconnect (&optional kill-daemon)
  "Close the RPC connection.
With KILL-DAEMON also shut down a daemon that this Emacs started."
  (interactive "P")
  (let ((own (and kill-daemon twisty--daemon-process
                  (process-live-p twisty--daemon-process))))
    (when (and own (twisty--live-p))
      (ignore-errors (twisty-request-sync "shutdown" nil 3)))
    (when (twisty--live-p)
      (delete-process twisty--connection))
    (setq twisty--connection nil)
    (when own
      (let ((deadline (+ (float-time) 3)))
        (while (and (process-live-p twisty--daemon-process) (< (float-time) deadline))
          (accept-process-output twisty--daemon-process 0.05)))
      (when (process-live-p twisty--daemon-process)
        (delete-process twisty--daemon-process))
      (setq twisty--daemon-process nil)
      (ignore-errors (delete-file twisty-socket-path)))))

(defun twisty-request (method &optional params callback)
  "Send METHOD with PARAMS (an alist) to the daemon.
CALLBACK, if given, is called with (RESULT ERROR).  Return the request id."
  (twisty-connect)
  (let ((id (cl-incf twisty--next-id)))
    (when callback
      (puthash id callback twisty--callbacks))
    (process-send-string
     twisty--connection
     (concat (json-serialize `((jsonrpc . "2.0") (id . ,id) (method . ,method)
                               (params . ,(or params (make-hash-table)))))
             "\n"))
    id))

(defun twisty-request-sync (method &optional params timeout)
  "Call METHOD with PARAMS and wait for the result (up to TIMEOUT seconds)."
  (let* ((done nil) (result nil) (err nil)
         (id (twisty-request method params
                             (lambda (res e) (setq result res err e done t))))
         (deadline (+ (float-time) (or timeout twisty-request-timeout))))
    (while (and (not done) (twisty--live-p) (< (float-time) deadline))
      (accept-process-output twisty--connection 0.05))
    (cond (err (error "twisty: %s" (alist-get 'message err)))
          ((not done) (remhash id twisty--callbacks) (error "twisty: `%s' timed out" method)))
    result))

;;;; Rendering -----------------------------------------------------------------------------------

(defun twisty--row-string (row)
  "Convert ROW, a list of (TEXT FG BG) runs, into a propertized string."
  (mapconcat
   (lambda (run)
     (let ((text (nth 0 run)) (fg (nth 1 run)) (bg (nth 2 run)))
       (if (or fg bg)
           (propertize text 'face (append (and fg (list :foreground fg))
                                          (and bg (list :background bg))))
         text)))
   row ""))

(defun twisty--frame-lines (frame)
  "Return the rows of FRAME (a `render' result) as propertized strings."
  (mapcar #'twisty--row-string (alist-get 'rows frame)))

(defun twisty--sparkline (values width)
  "Render the last WIDTH numbers of VALUES as a block sparkline."
  (let* ((vals (last values width))
         (lo (and vals (apply #'min vals)))
         (hi (and vals (apply #'max vals)))
         (span (if (and vals (> hi lo)) (- hi lo) 1.0)))
    (mapconcat (lambda (v)
                 (string (aref "▁▂▃▄▅▆▇█" (min 7 (max 0 (floor (* 7.999 (/ (- v lo) span))))))))
               vals "")))

(defun twisty--pad (string width)
  "Truncate or pad STRING to exactly WIDTH columns."
  (truncate-string-to-width string width nil ?\s))

;;;; Corner widget -------------------------------------------------------------------------------

(defvar twisty--corner-buffer nil "Buffer currently decorated with the corner widget.")
(defvar twisty--corner-overlays nil)
(defvar twisty--timer nil)
(defvar twisty--pending nil "Non-nil while a dashboard request is in flight.")
(defvar twisty--pending-since 0.0)

(defun twisty--corner-clear ()
  "Remove the corner widget overlays."
  (mapc #'delete-overlay twisty--corner-overlays)
  (setq twisty--corner-overlays nil))

(defun twisty--corner-paint (lines)
  "Paint LINES into the top-right corner of `twisty--corner-buffer'."
  (when (buffer-live-p twisty--corner-buffer)
    (with-current-buffer twisty--corner-buffer
      (twisty--corner-clear)
      (let* ((width (car twisty-corner-size))
             (lead (propertize " " 'display `(space :align-to (- right ,(1+ width)))))
             (win (get-buffer-window twisty--corner-buffer t))
             (start (if win (window-start win) (point-min)))
             (rest lines))
        (save-excursion
          (goto-char start)
          (while (and rest (not (eobp)))
            (let ((ov (make-overlay (line-end-position) (line-end-position))))
              (overlay-put ov 'twisty t)
              (overlay-put ov 'after-string (concat lead (car rest)))
              (push ov twisty--corner-overlays))
            (setq rest (cdr rest))
            (forward-line 1))
          (when rest
            ;; the buffer has fewer lines than the widget: hang the remainder below its end
            (goto-char (point-max))
            (let ((ov (make-overlay (point-max) (point-max))))
              (overlay-put ov 'twisty t)
              (overlay-put ov 'after-string
                           (concat (if (bolp) "" "\n")
                                   (mapconcat (lambda (line) (concat lead line)) rest "\n")))
              (push ov twisty--corner-overlays))))))))

;;;###autoload
(defun twisty-dashboard-attach (&optional buffer)
  "Show the rotating puzzle in the top-right corner of BUFFER (default: current)."
  (interactive)
  (twisty--corner-clear)
  (setq twisty--corner-buffer (or buffer (window-buffer (selected-window))))
  (twisty--ensure-timer)
  (twisty--tick))

(defun twisty-dashboard-detach ()
  "Remove the corner widget."
  (interactive)
  (when (buffer-live-p twisty--corner-buffer)
    (with-current-buffer twisty--corner-buffer (twisty--corner-clear)))
  (setq twisty--corner-buffer nil))

;;;; Dashboard buffer ----------------------------------------------------------------------------

(defconst twisty-buffer-name "*twisty*")
(defvar twisty--status nil "Last status alist received from the daemon.")
(defvar twisty--message "" "Transient message shown in the dashboard.")
(defvar twisty--show-net t)
(defvar twisty--playback nil "Moves still to be played back.")
(defvar twisty--playback-timer nil)
(defvar twisty--group-info nil)
(defvar twisty--macros nil)

(defun twisty--kv (key value)
  "Format a KEY / VALUE dashboard line."
  (concat (propertize (format "%-11s" key) 'face 'twisty-key) (format "%s" value)))

(defun twisty--left-column (status telemetry)
  "Build the text column of the dashboard from STATUS and TELEMETRY."
  (let* ((scalars (alist-get 'scalars telemetry))
         (series (alist-get 'series telemetry))
         (loss (alist-get 'loss series))
         (events (alist-get 'events telemetry))
         (solution (alist-get 'last_solution status))
         (lines
          (list
           (propertize (format "TwistyRL · %s" (alist-get 'puzzle status)) 'face 'twisty-title)
           (propertize (or (alist-get 'description status) "") 'face 'twisty-dim)
           ""
           (twisty--kv "state" (if (alist-get 'solved status)
                                   (propertize "solved" 'face 'success)
                                 (propertize "scrambled" 'face 'warning)))
           (twisty--kv "moves" (format "%d   %s" (alist-get 'moves status)
                                       (string-join (last (alist-get 'history status) 12) " ")))
           (twisty--kv "|G| (BSGS)" (or (alist-get 'group_order status) "…"))
           (twisty--kv "device" (alist-get 'device status))
           (twisty--kv "training"
                       (format "%s  step %s"
                               (if (alist-get 'training status) (alist-get 'training_mode status) "idle")
                               (or (alist-get 'train_step scalars) 0)))
           (twisty--kv "loss" (if loss
                                  (format "%.4f %s" (car (last loss)) (twisty--sparkline loss 28))
                                "—"))
           (twisty--kv "search"
                       (if (alist-get 'search_nodes_per_sec scalars)
                           (format "%.0f nodes/s  depth %.0f"
                                   (alist-get 'search_nodes_per_sec scalars)
                                   (or (alist-get 'search_depth scalars) 0))
                         "—"))
           (twisty--kv "solution" (if solution (string-join solution " ") "—"))
           "")))
    (when twisty--macros
      (setq lines (append lines (list (propertize "commutator macros" 'face 'twisty-key))))
      (dolist (m (seq-take twisty--macros 5))
        (setq lines (append lines (list (format "  %s  [%s, supp %s]" (alist-get 'text m)
                                                (alist-get 'kind m) (alist-get 'support m))))))
      (setq lines (append lines (list ""))))
    (when twisty--group-info
      (setq lines (append lines
                          (list (propertize "stabiliser chain" 'face 'twisty-key)
                                (format "  base %s" (alist-get 'base twisty--group-info))
                                (format "  orbits %s" (alist-get 'orbit_sizes twisty--group-info))
                                ""))))
    (setq lines (append lines (list (propertize "events" 'face 'twisty-key))))
    (dolist (e (last events 5))
      (setq lines (append lines (list (propertize (concat "  " e) 'face 'twisty-dim)))))
    (append lines
            (list ""
                  (unless (string-empty-p twisty--message) (propertize twisty--message 'face 'warning))
                  (propertize "U D L R F B turn · lowercase = inverse · m moves · s scramble · SPC solve"
                              'face 'twisty-dim)
                  (propertize "z undo · 0 reset · p puzzle · g BSGS · c macros · t train · a PAIRED"
                              'face 'twisty-dim)
                  (propertize "arrows rotate · . spin · + - zoom · n net · q quit"
                              'face 'twisty-dim)))))

(defun twisty--paint-dashboard (data)
  "Redraw the dashboard buffer from DATA, a `dashboard' RPC result."
  (let ((buf (get-buffer twisty-buffer-name)))
    (when (buffer-live-p buf)
      (with-current-buffer buf
        (let* ((inhibit-read-only t)
               (pt (point))
               (status (alist-get 'status data))
               (frame-lines (twisty--frame-lines (alist-get 'frame data)))
               (left (delq nil (twisty--left-column status (alist-get 'telemetry data))))
               (win (get-buffer-window buf t))
               (total (if win (window-body-width win) 100))
               (left-width (max 30 (- total (car twisty-frame-size) 2)))
               (n (max (length left) (length frame-lines))))
          (setq twisty--status status)
          (erase-buffer)
          (dotimes (i n)
            (insert (twisty--pad (or (nth i left) "") left-width) " "
                    (or (nth i frame-lines) "") "\n"))
          (when (and twisty--show-net (alist-get 'net data))
            (insert "\n")
            (dolist (line (twisty--frame-lines (alist-get 'net data)))
              (insert line "\n")))
          (goto-char (min pt (point-max))))))))

(defun twisty--dashboard-params ()
  "Parameters of the periodic `dashboard' request."
  (let* ((buf (get-buffer twisty-buffer-name))
         (win (and buf (get-buffer-window buf t))))
    (if win
        `((width . ,(car twisty-frame-size)) (height . ,(cdr twisty-frame-size))
          (net_width . ,(if twisty--show-net (max 20 (min 90 (- (window-body-width win) 2))) 0))
          (net_height . ,(if twisty--show-net 14 0)))
      `((width . ,(car twisty-corner-size)) (height . ,(cdr twisty-corner-size))))))

(defun twisty--tick ()
  "Request one refresh of every visible TwistyRL view."
  (when (and twisty--pending (> (- (float-time) twisty--pending-since) 5.0))
    (setq twisty--pending nil))
  (let* ((buf (get-buffer twisty-buffer-name))
         (dash (and buf (get-buffer-window buf t)))
         (corner (and (buffer-live-p twisty--corner-buffer)
                      (get-buffer-window twisty--corner-buffer t))))
    (when (and (or dash corner) (not twisty--pending))
      (condition-case err
          (progn
            (setq twisty--pending t twisty--pending-since (float-time))
            (twisty-request
             "dashboard" (twisty--dashboard-params)
             (lambda (result error)
               (setq twisty--pending nil)
               (when (and result (not error))
                 (setq twisty--status (alist-get 'status result))
                 (when (get-buffer-window twisty-buffer-name t)
                   (twisty--paint-dashboard result))
                 (when (and (buffer-live-p twisty--corner-buffer)
                            (not (eq twisty--corner-buffer (get-buffer twisty-buffer-name))))
                   (if (get-buffer-window twisty-buffer-name t)
                       ;; the corner widget needs its own, smaller frame
                       (twisty-request
                        "render" `((width . ,(car twisty-corner-size)) (height . ,(cdr twisty-corner-size)))
                        (lambda (frame err2)
                          (when (and frame (not err2))
                            (twisty--corner-paint (twisty--frame-lines frame)))))
                     (twisty--corner-paint (twisty--frame-lines (alist-get 'frame result)))))))))
        (error
         (setq twisty--pending nil)
         (message "twisty: %s" (error-message-string err)))))))

(defun twisty--ensure-timer ()
  "Start the refresh timer if it is not running."
  (unless (and twisty--timer (memq twisty--timer timer-list))
    (setq twisty--timer (run-with-timer 0 (/ 1.0 (max 1 twisty-fps)) #'twisty--tick))))

(defun twisty--stop-timer ()
  "Stop the refresh timer."
  (when twisty--timer
    (cancel-timer twisty--timer)
    (setq twisty--timer nil)))

;;;; Interactive commands ------------------------------------------------------------------------

(defun twisty--call (method &optional params)
  "Synchronously call METHOD with PARAMS, reporting errors in the dashboard."
  (condition-case err
      (prog1 (twisty-request-sync method params)
        (twisty--tick))
    (error (setq twisty--message (error-message-string err))
           (message "%s" twisty--message)
           nil)))

(defun twisty--actions ()
  "Action names of the current puzzle."
  (alist-get 'actions (or twisty--status (setq twisty--status (twisty-request-sync "status")))))

(defun twisty-move (moves)
  "Apply MOVES, a whitespace separated move string such as \"R U R' U'\"."
  (interactive
   (list (completing-read-multiple "Moves (comma or space separated): " (twisty--actions))))
  (let ((text (if (listp moves) (string-join moves " ") moves)))
    (setq twisty--message "")
    (twisty--call "apply" `((moves . ,text)))))

(defun twisty-key-move ()
  "Turn the face named by the key that invoked this command.
An uppercase key turns clockwise.  The lowercase key turns
counter-clockwise, unless the puzzle has a move with that lowercase
name (Pyraminx tips)."
  (interactive)
  (let* ((key (string last-command-event))
         (actions (twisty--actions))
         (move (cond ((member key actions) key)
                     ((member (concat (upcase key) "'") actions) (concat (upcase key) "'"))
                     ((member (upcase key) actions) (upcase key)))))
    (if move
        (twisty-move move)
      (setq twisty--message (format "this puzzle has no move `%s' (press m for the list)" key))
      (message "%s" twisty--message))))

(defun twisty-scramble (&optional depth)
  "Scramble the puzzle with DEPTH random moves (prefix argument, default 20)."
  (interactive "P")
  (twisty--stop-playback)
  (let ((res (twisty--call "scramble" `((depth . ,(if depth (prefix-numeric-value depth) 20))))))
    (when res
      (setq twisty--message (format "scramble: %s" (string-join (alist-get 'scramble res) " "))))))

(defun twisty-reset ()
  "Return the puzzle to its solved state."
  (interactive)
  (twisty--stop-playback)
  (twisty--call "reset"))

(defun twisty-undo ()
  "Undo the last move."
  (interactive)
  (twisty--call "undo"))

(defun twisty--stop-playback ()
  "Abort a running solution playback."
  (when twisty--playback-timer
    (cancel-timer twisty--playback-timer))
  (setq twisty--playback nil twisty--playback-timer nil))

(defun twisty--playback-step ()
  "Apply the next move of the solution being played back."
  (if (null twisty--playback)
      (twisty--stop-playback)
    (let ((move (pop twisty--playback)))
      (twisty-request "apply" `((moves . ,move)) (lambda (_r _e) (twisty--tick))))))

(defun twisty-solve (&optional method)
  "Ask the search engine for a solution and play it back.
With a prefix argument prompt for the search METHOD."
  (interactive
   (list (if current-prefix-arg (completing-read "Method: " '("auto" "astar" "beam") nil t) "auto")))
  (twisty--stop-playback)
  (setq twisty--message "searching…")
  (twisty-request
   "solve" `((method . ,(or method "auto")))
   (lambda (result error)
     (cond (error (setq twisty--message (format "solve failed: %s" (alist-get 'message error))))
           ((alist-get 'solved result)
            (let ((moves (alist-get 'moves result)))
              (setq twisty--message
                    (format "solved in %d moves%s · %d nodes · %.2fs"
                            (length moves)
                            (if (alist-get 'optimal_proven result) " (optimal)" "")
                            (alist-get 'nodes_generated result) (alist-get 'elapsed result)))
              (setq twisty--playback (copy-sequence moves))
              (setq twisty--playback-timer
                    (run-with-timer 0.05 twisty-playback-delay #'twisty--playback-step))))
           (t (setq twisty--message (format "no solution: %s" (alist-get 'reason result)))))
     (message "%s" twisty--message))))

(defun twisty-select-puzzle (name)
  "Switch the daemon to puzzle NAME."
  (interactive
   (list (completing-read "Puzzle: " (twisty-request-sync "list_puzzles") nil t)))
  (twisty--stop-playback)
  (setq twisty--group-info nil twisty--macros nil twisty--message "")
  (twisty--call "set_puzzle" `((name . ,name))))

(defun twisty-group-info ()
  "Run Schreier-Sims on the current puzzle and show the stabiliser chain."
  (interactive)
  (setq twisty--message "running Schreier-Sims…")
  (twisty-request
   "group_info" nil
   (lambda (result error)
     (if error
         (setq twisty--message (alist-get 'message error))
       (setq twisty--group-info result
             twisty--message (format "|G| = %s · base length %d · %s"
                                     (alist-get 'order_text result)
                                     (length (alist-get 'base result))
                                     (if (alist-get 'solvable result) "state solvable" "STATE UNSOLVABLE"))))
     (message "%s" twisty--message))))

(defun twisty-mine-macros (&optional promote)
  "Mine commutator and conjugate macros.
With prefix PROMOTE add that many of the best to the action space."
  (interactive "P")
  (setq twisty--message "mining commutators…")
  (twisty-request
   "mine_macros" `((top_k . 8) (promote . ,(if promote (prefix-numeric-value promote) 0)))
   (lambda (result error)
     (if error
         (setq twisty--message (alist-get 'message error))
       (setq twisty--macros result
             twisty--message (format "mined %d macros" (length result))))
     (message "%s" twisty--message))))

(defun twisty-train (&optional adversarial)
  "Toggle background ADI training; with prefix ADVERSARIAL run PAIRED self-play."
  (interactive "P")
  (if (alist-get 'training twisty--status)
      (progn (twisty--call "train_stop") (setq twisty--message "training stopped"))
    (twisty--call "train" `((steps . ,(if adversarial 150 400))
                            (adversarial . ,(if adversarial t :false))))
    (setq twisty--message (if adversarial "PAIRED self-play started" "ADI training started"))))

(defun twisty-train-adversarial ()
  "Start PAIRED adversarial self-play."
  (interactive)
  (twisty-train t))

(defun twisty-rotate-left () "Rotate the camera left." (interactive)
       (twisty--call "rotate" '((dyaw . -0.3))))
(defun twisty-rotate-right () "Rotate the camera right." (interactive)
       (twisty--call "rotate" '((dyaw . 0.3))))
(defun twisty-rotate-up () "Tilt the camera up." (interactive)
       (twisty--call "rotate" '((dpitch . -0.2))))
(defun twisty-rotate-down () "Tilt the camera down." (interactive)
       (twisty--call "rotate" '((dpitch . 0.2))))
(defun twisty-zoom-in () "Zoom in." (interactive) (twisty--call "rotate" '((zoom . 1.1))))
(defun twisty-zoom-out () "Zoom out." (interactive) (twisty--call "rotate" '((zoom . 0.9))))
(defun twisty-toggle-spin () "Toggle the automatic rotation." (interactive) (twisty--call "spin"))

(defun twisty-toggle-net ()
  "Toggle the 2D net view."
  (interactive)
  (setq twisty--show-net (not twisty--show-net))
  (twisty--tick))

(defun twisty-quit ()
  "Bury the dashboard buffer."
  (interactive)
  (twisty--stop-playback)
  (quit-window))

(defvar twisty-mode-map
  (let ((map (make-sparse-keymap)))
    (suppress-keymap map t)
    (dolist (key '("U" "D" "L" "R" "F" "B" "u" "d" "l" "r" "f" "b"))
      (define-key map (kbd key) #'twisty-key-move))
    (define-key map (kbd "m") #'twisty-move)
    (define-key map (kbd "s") #'twisty-scramble)
    (define-key map (kbd "SPC") #'twisty-solve)
    (define-key map (kbd "x") #'twisty-solve)
    (define-key map (kbd "0") #'twisty-reset)
    (define-key map (kbd "z") #'twisty-undo)
    (define-key map (kbd "p") #'twisty-select-puzzle)
    (define-key map (kbd "g") #'twisty-group-info)
    (define-key map (kbd "c") #'twisty-mine-macros)
    (define-key map (kbd "t") #'twisty-train)
    (define-key map (kbd "a") #'twisty-train-adversarial)
    (define-key map (kbd "n") #'twisty-toggle-net)
    (define-key map (kbd ".") #'twisty-toggle-spin)
    (define-key map (kbd "<left>") #'twisty-rotate-left)
    (define-key map (kbd "<right>") #'twisty-rotate-right)
    (define-key map (kbd "<up>") #'twisty-rotate-up)
    (define-key map (kbd "<down>") #'twisty-rotate-down)
    (define-key map (kbd "+") #'twisty-zoom-in)
    (define-key map (kbd "=") #'twisty-zoom-in)
    (define-key map (kbd "-") #'twisty-zoom-out)
    (define-key map (kbd "q") #'twisty-quit)
    map)
  "Keymap of `twisty-mode'.")

(define-derived-mode twisty-mode special-mode "Twisty"
  "Major mode of the TwistyRL dashboard buffer.

\\{twisty-mode-map}"
  (setq-local truncate-lines t)
  (setq-local cursor-type nil)
  (setq-local line-spacing 0)
  (buffer-disable-undo))

;;;###autoload
(defun twisty-dashboard ()
  "Open the interactive TwistyRL dashboard."
  (interactive)
  (twisty-connect)
  (let ((buf (get-buffer-create twisty-buffer-name)))
    (with-current-buffer buf
      (unless (derived-mode-p 'twisty-mode)
        (twisty-mode)))
    (pop-to-buffer-same-window buf)
    (twisty--paint-dashboard
     (twisty-request-sync "dashboard" (twisty--dashboard-params)))
    (twisty--ensure-timer)
    buf))

;;;; Global minor mode ---------------------------------------------------------------------------

(defun twisty--attach-startup ()
  "Attach the corner widget to the buffer shown at startup."
  (condition-case err
      (progn
        (twisty-connect)
        (twisty-dashboard-attach (window-buffer (selected-window))))
    (error (message "twisty-dashboard: %s" (error-message-string err)))))

(defun twisty--shutdown ()
  "Stop timers and a daemon owned by this Emacs."
  (twisty--stop-timer)
  (twisty--stop-playback)
  (ignore-errors (twisty-disconnect t)))

;;;###autoload
(define-minor-mode twisty-dashboard-mode
  "Show a rotating 3D TwistyRL puzzle in the corner of the startup buffer.
The daemon is started on demand.
\\[twisty-dashboard] opens the full interactive dashboard."
  :global t
  :lighter " Twisty"
  (if twisty-dashboard-mode
      (progn
        (add-hook 'kill-emacs-hook #'twisty--shutdown)
        (if after-init-time
            (run-with-idle-timer 0.2 nil #'twisty--attach-startup)
          (add-hook 'emacs-startup-hook
                    (lambda () (run-with-idle-timer 0.3 nil #'twisty--attach-startup)))))
    (twisty-dashboard-detach)
    (unless (get-buffer-window twisty-buffer-name t)
      (twisty--stop-timer))))

;;;; Batch self-test -----------------------------------------------------------------------------

(defun twisty-dashboard-self-test ()
  "Exercise the client against a running daemon (used by the tests).
The socket is taken from the environment variable TWISTY_SOCKET."
  (let ((twisty-socket-path (or (getenv "TWISTY_SOCKET") twisty-socket-path)))
    (twisty-connect t)
    (cl-assert (alist-get 'pong (twisty-request-sync "ping")))
    (cl-assert (member "megaminx" (twisty-request-sync "list_puzzles")))
    (twisty-request-sync "set_puzzle" '((name . "cube3")))
    (twisty-request-sync "reset")
    (let ((res (twisty-request-sync "apply" '((moves . "R U R' U'")))))
      (cl-assert (equal (alist-get 'applied res) '("R" "U" "R'" "U'")))
      (cl-assert (not (alist-get 'solved res))))
    (let* ((frame (twisty-request-sync "render" '((width . 30) (height . 15))))
           (lines (twisty--frame-lines frame)))
      (cl-assert (= (length lines) 15))
      (cl-assert (cl-some (lambda (l) (string-match-p "[▀▄]" l)) lines))
      (cl-assert (cl-some (lambda (l) (text-property-not-all 0 (length l) 'face nil l)) lines))
      ;; corner widget on a short buffer
      (with-temp-buffer
        (insert "hello\nworld\n")
        (setq twisty--corner-buffer (current-buffer))
        (twisty--corner-paint lines)
        (cl-assert (>= (length twisty--corner-overlays) 3))
        (cl-assert (cl-every (lambda (ov) (overlay-get ov 'after-string)) twisty--corner-overlays))
        (twisty--corner-clear)
        (setq twisty--corner-buffer nil)))
    ;; full dashboard buffer
    (let ((buf (get-buffer-create twisty-buffer-name)))
      (with-current-buffer buf (twisty-mode))
      (twisty--paint-dashboard
       (twisty-request-sync "dashboard" '((width . 40) (height . 20) (net_width . 60) (net_height . 12))))
      (with-current-buffer buf
        (cl-assert (string-match-p "TwistyRL · cube3" (buffer-string)))
        (cl-assert (string-match-p "scrambled" (buffer-string)))
        (cl-assert (> (count-lines (point-min) (point-max)) 30))))
    ;; solve the four-move scramble and replay the solution
    (let ((res (twisty-request-sync "solve" '((time_limit . 30)))))
      (cl-assert (alist-get 'solved res))
      (twisty-request-sync "apply" `((moves . ,(string-join (alist-get 'moves res) " "))))
      (cl-assert (alist-get 'solved (twisty-request-sync "status"))))
    (cl-assert (equal (twisty--sparkline '(0.0 1.0) 2) "▁█"))
    (cl-assert (eq (lookup-key twisty-mode-map (kbd "SPC")) #'twisty-solve))
    (twisty-disconnect)
    (princ "TWISTY-EMACS-OK\n")))

(provide 'twisty-dashboard)

;;; twisty-dashboard.el ends here
