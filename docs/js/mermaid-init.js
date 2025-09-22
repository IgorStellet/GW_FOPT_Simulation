/* docs/js/mermaid-init.js */
window.addEventListener("DOMContentLoaded", () => {
  if (window.mermaid) {
    window.mermaid.initialize({
      startOnLoad: true,      // render anything fenced as ```mermaid
      theme: "default"        // or "neutral", "dark", etc.
    });
  }
});
