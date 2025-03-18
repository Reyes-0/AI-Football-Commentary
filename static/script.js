document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("fileInput");
    const pathName = document.getElementById("path-name");

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) {
            pathName.textContent = fileInput.files[0].name;
        } else {
            pathName.textContent = "Please Select a file";
        }
    });
});