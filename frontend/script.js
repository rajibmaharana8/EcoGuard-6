
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const fileNameDisplay = document.getElementById('file-name-display');
const originalImg = document.getElementById('original-img');
const heatmapImg = document.getElementById('heatmap-img');
const loader = document.getElementById('loader');
const logFeed = document.getElementById('log-feed');
const statusPill = document.getElementById('status-pill');
const confidenceText = document.getElementById('confidence-text');
const confidenceBar = document.getElementById('confidence-bar');
const threatLevel = document.getElementById('threat-level');

function addLog(msg) {
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    const time = new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    entry.innerHTML = `<span style="opacity:0.5">[${time}]</span> ${msg}`;
    logFeed.innerHTML = ''; // Keep only the latest log for cleaner UI
    logFeed.appendChild(entry);
}

let userLocation = { lat: null, lng: null };

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--primary)';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = 'var(--glass-border)';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--glass-border)';
    if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        addLog('Error: Invalid file type');
        return;
    }
    fileNameDisplay.innerText = file.name;

    addLog('Requesting Geo-Precise location for report accuracy...');
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                userLocation.lat = pos.coords.latitude;
                userLocation.lng = pos.coords.longitude;
                addLog(`Geo-Tag Attached: ${userLocation.lat.toFixed(4)}, ${userLocation.lng.toFixed(4)}`);
                finishHandling(file);
            },
            () => {
                addLog('Warning: Location denied. Proceeding with standard scan.');
                finishHandling(file);
            }
        );
    } else {
        finishHandling(file);
    }
}

function finishHandling(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        originalImg.src = e.target.result;
    };
    reader.readAsDataURL(file);
    uploadAndPredict(file);
}

async function uploadAndPredict(file) {
    loader.classList.remove('hidden');
    heatmapImg.classList.add('hidden');

    const mode = document.getElementById('mode-select').value;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('lat', userLocation.lat);
    formData.append('lng', userLocation.lng);

    try {
        const startTime = Date.now();
        const response = await fetch(`/predict?mode=${mode}`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        const duration = ((Date.now() - startTime) / 1000).toFixed(2);

        if (data.success) {
            addLog(`Analysis complete (${duration}s)`);
            heatmapImg.src = data.heatmap;
            heatmapImg.classList.remove('hidden');
            confidenceText.innerText = `${data.confidence}%`;
            confidenceBar.style.width = `${data.confidence}%`;

            statusPill.innerText = data.prediction;
            statusPill.className = `status-pill status-${data.status_type}`;

            if (data.status_type === 'danger') {
                threatLevel.innerText = 'CRITICAL';
                threatLevel.style.color = 'var(--danger)';
            } else if (data.status_type === 'warning') {
                threatLevel.innerText = 'ELEVATED';
                threatLevel.style.color = 'var(--warning)';
            } else {
                threatLevel.innerText = 'LOW';
                threatLevel.style.color = 'var(--success)';
            }

            if (data.community_alert) {
                addLog('!!! COMMUNITY ALERT TRIGGERED !!! Notification sent to Municipality.', 'danger');
                threatLevel.innerText = 'COMMUNITY THREAT';
                threatLevel.style.color = 'var(--danger)';
                threatLevel.style.textShadow = '0 0 10px var(--danger)';
            }
        } else {
            addLog('System Error: ' + data.error);
        }
    } catch (error) {
        addLog('Network Error: Check backend connectivity.');
        console.error(error);
    } finally {
        loader.classList.add('hidden');
    }
}
