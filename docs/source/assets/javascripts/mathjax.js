window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
  },
  options: {
    ignoreHtmlClass: "\\b(?:no-mathjax)\\b",
    processHtmlClass: "\\b(?:arithmatex)\\b",
  },
};
