"use strict";

const menuButton = document.getElementById("menu-toggle");
const sidebar = document.getElementById("sidebar");

if (menuButton && sidebar) {
  menuButton.addEventListener("click", () => {
    const isOpen = sidebar.classList.toggle("open");
    menuButton.setAttribute("aria-expanded", String(isOpen));
  });

  document.addEventListener("click", (event) => {
    if (window.innerWidth <= 680 && sidebar.classList.contains("open") &&
        !sidebar.contains(event.target) && !menuButton.contains(event.target)) {
      sidebar.classList.remove("open");
      menuButton.setAttribute("aria-expanded", "false");
    }
  });
}
