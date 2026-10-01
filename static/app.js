(function () {
  var panel = document.getElementById("results");
  var backdrop = document.getElementById("results-backdrop");
  var showBtn = document.getElementById("show-results");
  var form = document.querySelector("form[data-ajax]");
  if (!panel || !form) return;

  var mobile = window.matchMedia("(max-width: 960px)");
  var hasResult = panel.getAttribute("data-has-result") === "1";

  function setOpen(open) {
    if (!mobile.matches) {
      backdrop.hidden = true;
      showBtn.hidden = true;
      panel.classList.remove("open");
      return;
    }
    panel.classList.toggle("open", open);
    backdrop.hidden = !open;
    showBtn.hidden = open || !hasResult;
  }

  function showHtml(html, ok) {
    panel.innerHTML = html;
    hasResult = true;
    panel.setAttribute("data-has-result", "1");
    panel.scrollTop = 0;
    setOpen(true);
    bindClose();
    return ok;
  }

  function errorHtml(message) {
    var wrap = document.createElement("div");
    var head = document.createElement("div");
    head.className = "panel-head";
    head.innerHTML = '<h2>Results</h2><button type="button" class="close-btn" data-close aria-label="Close results">Close ✕</button>';
    var alert = document.createElement("div");
    alert.className = "alert error";
    alert.textContent = message;
    wrap.appendChild(head);
    wrap.appendChild(alert);
    return wrap.innerHTML;
  }

  function bindClose() {
    var btn = panel.querySelector("[data-close]");
    if (btn) btn.addEventListener("click", function () { setOpen(false); });
  }

  form.addEventListener("submit", function (event) {
    if (!window.fetch || !window.FormData) return;
    if (!form.checkValidity()) return;
    event.preventDefault();

    var submit = form.querySelector('button[type="submit"]');
    var label = submit.textContent;
    submit.disabled = true;
    submit.textContent = "Predicting…";
    panel.classList.add("loading");

    var controller = window.AbortController ? new AbortController() : null;
    var timer = controller ? setTimeout(function () { controller.abort(); }, 90000) : null;

    fetch(form.getAttribute("action") || window.location.pathname, {
      method: "POST",
      body: new FormData(form),
      headers: { "X-Requested-With": "fetch" },
      signal: controller ? controller.signal : undefined
    })
      .then(function (response) {
        return response.text().then(function (text) {
          if (response.headers.get("X-Result-Panel") === "1") {
            showHtml(text, response.ok);
          } else {
            showHtml(errorHtml("The server did not return a result (HTTP " + response.status + "). Please try again in a moment."), false);
          }
        });
      })
      .catch(function () {
        showHtml(errorHtml("The request could not be completed. Please check your connection and try again."), false);
      })
      .then(function () {
        if (timer) clearTimeout(timer);
        submit.disabled = false;
        submit.textContent = label;
        panel.classList.remove("loading");
      });
  });

  backdrop.addEventListener("click", function () { setOpen(false); });
  showBtn.addEventListener("click", function () { setOpen(true); });
  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") setOpen(false);
  });
  mobile.addEventListener ? mobile.addEventListener("change", function () { setOpen(false); }) : null;

  bindClose();
  if (hasResult) setOpen(true);
})();
