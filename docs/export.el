;; Export docs/orgmode to docs/source for Sphinx.
(require 'package)
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/") t)
(package-initialize)

(unless (package-installed-p 'ox-rst)
  (package-refresh-contents)
  (package-install 'ox-rst))

(require 'ox-rst)
(require 'ox-publish)

;; ox-rst uses this Org 9.7 predicate. Keep the exporter working on
;; distributions that still package Org 9.6.
(unless (fboundp 'org-element-type-p)
  (defun org-element-type-p (element type)
    (eq (org-element-type element) type)))

(unless (fboundp 'org-element-parent)
  (defun org-element-parent (element)
    (org-element-property :parent element)))

(unless (fboundp 'org-element-parent-element)
  (defun org-element-parent-element (element)
    (let ((parent (org-element-parent element)))
      (while (and parent
                  (not (memq (org-element-type parent)
                             org-element-all-elements)))
        (setq parent (org-element-parent parent)))
      parent)))

(unless (fboundp 'org-element-extract)
  (defun org-element-extract (element)
    (let ((parent (org-element-parent element)))
      (when parent
        (setcdr (cdr parent) (delq element (cddr parent))))
      element)))

(setq org-publish-project-alist
      '(("optbench"
         :base-directory "./docs/orgmode/"
         :base-extension "org"
         :publishing-directory "./docs/source/"
         :publishing-function org-rst-publish-to-rst
         :recursive t
         :headline-levels 3)))

(org-publish "optbench" t)
