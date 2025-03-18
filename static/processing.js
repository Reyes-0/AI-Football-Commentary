document.addEventListener("DOMContentLoaded", () => {
    function checkStatus() {
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                if (data.completed) {
                    document.getElementById('status').textContent = "Video is ready to download";
                    const downloadButton = document.createElement("button");
                    downloadButton.textContent = "Download File";
                    downloadButton.onclick = () => window.location.href = '/download';
                    document.querySelector('.container').appendChild(downloadButton);
                } else {
                    setTimeout(checkStatus, 1000);
                }
            })
            .catch(error => console.error('Error:', error));
    }

    checkStatus();
});