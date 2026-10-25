/* Applies the saved colour theme before the first paint, so there is no flash of the wrong theme.
   Loaded as a small blocking script in <head> on purpose (the page's CSP forbids inline scripts).
   The switch itself is wired up in app.js. No saved choice = follow the operating system. */
(function () {
  try {
    var saved = localStorage.getItem("verbadb.theme");
    if (saved === "light" || saved === "dark") {
      document.documentElement.setAttribute("data-theme", saved);
    }
  } catch (e) { /* storage blocked (private mode): follow the system */ }
})();
