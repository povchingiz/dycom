/**
 * FaceSim Demo - Frontend Application
 * Handles login, upload, progress tracking, and downloads
 */

// ============== Translations ==============

const translations = {
    en: {
        login_title: "Enter Password",
        password_placeholder: "Password",
        login_btn: "Login",
        logout: "Logout",
        upload_title: "Upload DICOM Scan",
        upload_description: "Upload a CBCT DICOM file for automatic teeth and soft tissue segmentation.",
        upload_dropzone: "Drag & drop your DICOM file here, or <strong>click to browse</strong>",
        upload_hint: "Accepted format: .dcm (max 500MB)",
        scenario_title: "Surgical plan",
        scenario_hint: "Mandible movement to simulate. Leave the defaults for a standard 5 mm advancement.",
        scenario_advance: "Advancement, mm",
        scenario_vertical: "Vertical, mm",
        scenario_lateral: "Lateral, mm",
        scenario_pitch: "Pitch, \u00b0",
        scenario_simulate: "Predict the post-op face",
        results_sim: "Predicted face:",
        results_sim_none: "segmentation only, no prediction requested",
        processing_title: "Processing Your Scan",
        step_converting: "Converting DICOM to NIfTI...",
        step_teeth: "Segmenting teeth and jawbones...",
        step_soft: "Segmenting soft tissue...",
        step_meshes: "Generating 3D meshes...",
        step_package: "Preparing download package...",
        processing_note: "This process takes approximately 10-15 minutes on GPU.",
        results_title: "Processing Complete!",
        results_message: "Your segmentation results are ready for download.",
        results_files: "Files generated:",
        results_format: "Format:",
        results_session: "Session ID:",
        download_btn: "⬇ Download Results",
        new_scan_btn: "Process Another Scan",
        delete_session_btn: "Delete This Session",
        error_title: "Processing Failed",
        retry_btn: "Try Again",
        sessions_title: "Active Sessions",
        refresh_btn: "Refresh",
        footer_text: "FaceSim Demo — Open-source face simulation pipeline",
        status_created: "Created",
        status_queued: "Queued",
        status_processing: "Processing",
        status_completed: "Completed",
        status_failed: "Failed",
        status_downloaded: "Downloaded",
    },
    ru: {
        login_title: "Введите пароль",
        password_placeholder: "Пароль",
        login_btn: "Войти",
        logout: "Выйти",
        upload_title: "Загрузить DICOM скан",
        upload_description: "Загрузите файл CBCT DICOM для автоматической сегментации зубов и мягких тканей.",
        upload_dropzone: "Перетащите файл DICOM сюда или <strong>нажмите для выбора</strong>",
        upload_hint: "Формат: .dcm (макс. 500МБ)",
        scenario_title: "План операции",
        scenario_hint: "Смещение нижней челюсти для моделирования. По умолчанию — выдвижение на 5 мм.",
        scenario_advance: "Выдвижение, мм",
        scenario_vertical: "По вертикали, мм",
        scenario_lateral: "Вбок, мм",
        scenario_pitch: "Наклон, \u00b0",
        scenario_simulate: "Спрогнозировать лицо после операции",
        results_sim: "Прогноз лица:",
        results_sim_none: "только сегментация, прогноз не запрашивался",
        processing_title: "Обработка скана",
        step_converting: "Конвертация DICOM в NIfTI...",
        step_teeth: "Сегментация зубов и челюстей...",
        step_soft: "Сегментация мягких тканей...",
        step_meshes: "Генерация 3D сеток...",
        step_package: "Подготовка пакета для загрузки...",
        processing_note: "Процесс занимает примерно 10-15 минут на GPU.",
        results_title: "Обработка завершена!",
        results_message: "Результаты сегментации готовы к загрузке.",
        results_files: "Файлов создано:",
        results_format: "Формат:",
        results_session: "ID сессии:",
        download_btn: "⬇ Скачать результаты",
        new_scan_btn: "Обработать другой скан",
        delete_session_btn: "Удалить эту сессию",
        error_title: "Ошибка обработки",
        retry_btn: "Попробовать снова",
        sessions_title: "Активные сессии",
        refresh_btn: "Обновить",
        footer_text: "FaceSim Demo — Открытый пайплайн симуляции лица",
        status_created: "Создано",
        status_queued: "В очереди",
        status_processing: "Обработка",
        status_completed: "Завершено",
        status_failed: "Ошибка",
        status_downloaded: "Скачано",
    },
    kk: {
        login_title: "Құпия сөзді енгізіңіз",
        password_placeholder: "Құпия сөз",
        login_btn: "Кіру",
        logout: "Шығу",
        upload_title: "DICOM скан жүктеу",
        upload_description: "Тістер мен жұмсақ тіндерді автоматты сегментациялау үшін CBCT DICOM файлын жүктеңіз.",
        upload_dropzone: "DICOM файлын осы жерге сүйреңіз немесе <strong>таңдау үшін басыңыз</strong>",
        upload_hint: "Формат: .dcm (макс. 500МБ)",
        scenario_title: "Операция жоспары",
        scenario_hint: "Модельдеуге арналған төменгі жақ жылжуы. Әдепкі — 5 мм алға шығару.",
        scenario_advance: "Алға жылжу, мм",
        scenario_vertical: "Тігінен, мм",
        scenario_lateral: "Бүйірге, мм",
        scenario_pitch: "Көлбеу, \u00b0",
        scenario_simulate: "Операциядан кейінгі бетті болжау",
        results_sim: "Бет болжамы:",
        results_sim_none: "тек сегментация, болжам сұралмады",
        processing_title: "Сканды өңдеу",
        step_converting: "DICOM-ды NIfTI-ге түрлендіру...",
        step_teeth: "Тістер мен жақ сүйектерін сегментациялау...",
        step_soft: "Жұмсақ тіндерді сегментациялау...",
        step_meshes: "3D торларды генерациялау...",
        step_package: "Жүктеу пакетін дайындау...",
        processing_note: "Процесс GPU-да шамамен 10-15 минут алады.",
        results_title: "Өңдеу аяқталды!",
        results_message: "Сегментация нәтижелері жүктеуге дайын.",
        results_files: "Жасалған файлдар:",
        results_format: "Формат:",
        results_session: "Сессия ID:",
        download_btn: "⬇ Нәтижелерді жүктеу",
        new_scan_btn: "Басқа сканды өңдеу",
        delete_session_btn: "Бұл сессияны жою",
        error_title: "Өңдеу қатесі",
        retry_btn: "Қайта көру",
        sessions_title: "Белсенді сессиялар",
        refresh_btn: "Жаңарту",
        footer_text: "FaceSim Demo — Ашық бет симуляциясы пайплайны",
        status_created: "Жасалды",
        status_queued: "Кезекте",
        status_processing: "Өңдеу",
        status_completed: "Аяқталды",
        status_failed: "Қате",
        status_downloaded: "Жүктелді",
    },
};

let currentLang = 'en';

// ============== State ==============

let currentSessionId = null;
let progressPollInterval = null;

// ============== DOM Elements ==============

const elements = {
    loginSection: document.getElementById('login-section'),
    uploadSection: document.getElementById('upload-section'),
    processingSection: document.getElementById('processing-section'),
    resultsSection: document.getElementById('results-section'),
    errorSection: document.getElementById('error-section'),
    sessionsSection: document.getElementById('sessions-section'),
    
    loginForm: document.getElementById('login-form'),
    passwordInput: document.getElementById('password-input'),
    loginError: document.getElementById('login-error'),
    logoutBtn: document.getElementById('logout-btn'),
    
    uploadArea: document.getElementById('upload-area'),
    fileInput: document.getElementById('file-input'),
    uploadProgress: document.getElementById('upload-progress'),
    progressFill: document.getElementById('progress-fill'),
    progressText: document.getElementById('progress-text'),
    
    progressRing: document.getElementById('progress-ring'),
    progressPercentage: document.getElementById('progress-percentage'),
    
    downloadBtn: document.getElementById('download-btn'),
    newScanBtn: document.getElementById('new-scan-btn'),
    deleteSessionBtn: document.getElementById('delete-session-btn'),
    retryBtn: document.getElementById('retry-btn'),
    
    sessionIdDisplay: document.getElementById('session-id-display'),
    errorDetails: document.getElementById('error-details'),
    
    languageSelector: document.getElementById('language-selector'),
    themeToggle: document.getElementById('theme-toggle'),
};

// ============== Utility Functions ==============

function t(key) {
    return (translations[currentLang] && translations[currentLang][key])
        || translations.en[key]
        || key;
}

function updateTranslations() {
    const t = translations[currentLang];
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (t[key]) {
            el.innerHTML = t[key];
        }
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        const key = el.getAttribute('data-i18n-placeholder');
        if (t[key]) {
            el.placeholder = t[key];
        }
    });
    document.title = currentLang === 'en' ? 'FaceSim Demo' : 
                     currentLang === 'ru' ? 'FaceSim Демо' : 'FaceSim Демо';
}

function setTheme(isDark) {
    document.body.classList.toggle('dark-theme', isDark);
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    elements.themeToggle.querySelector('.theme-icon').textContent = isDark ? '☀️' : '🌙';
}

function toggleTheme() {
    const isDark = !document.body.classList.contains('dark-theme');
    setTheme(isDark);
}

function showSection(sectionId) {
    ['loginSection', 'uploadSection', 'processingSection', 'resultsSection', 'errorSection', 'sessionsSection'].forEach(id => {
        elements[id].style.display = 'none';
    });
    elements[sectionId].style.display = 'block';
}

function setProgress(percentage, message) {
    elements.progressFill.style.width = `${percentage}%`;
    elements.progressPercentage.textContent = `${percentage}%`;
    elements.progressText.textContent = message;
    
    // Update ring
    const circumference = 2 * Math.PI * 52;
    const offset = circumference - (percentage / 100) * circumference;
    elements.progressRing.style.strokeDashoffset = offset;
    
    // Update step indicators
    const step = Math.ceil(percentage / 20);
    for (let i = 1; i <= 5; i++) {
        const stepEl = document.getElementById(`step-${i}`);
        if (i < step) {
            stepEl.classList.add('completed');
            stepEl.querySelector('.step-icon').textContent = '✅';
        } else if (i === step) {
            stepEl.classList.add('active');
            stepEl.querySelector('.step-icon').textContent = '⚙️';
        } else {
            stepEl.classList.remove('active', 'completed');
            stepEl.querySelector('.step-icon').textContent = '⏳';
        }
    }
}

// ============== API Calls ==============

async function login(password) {
    const response = await fetch('/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Login failed');
    }
    
    return response.json();
}

async function logout() {
    await fetch('/logout', { method: 'POST' });
}

function readScenario() {
    const num = (id, fallback) => {
        const el = document.getElementById(id);
        const value = el ? parseFloat(el.value) : NaN;
        return Number.isFinite(value) ? value : fallback;
    };
    const toggle = document.getElementById('simulate-toggle');
    return {
        simulate: toggle ? toggle.checked : true,
        advance_mm: num('advance-mm', 5),
        vertical_mm: num('vertical-mm', 0),
        lateral_mm: num('lateral-mm', 0),
        pitch_deg: num('pitch-deg', 0),
    };
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const scenario = readScenario();
    Object.entries(scenario).forEach(([key, value]) => formData.append(key, value));

    const response = await fetch('/upload', {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Upload failed');
    }
    
    return response.json();
}

async function getSessionStatus(sessionId) {
    const response = await fetch(`/status/${sessionId}`);
    if (!response.ok) {
        throw new Error('Status check failed');
    }
    return response.json();
}

async function deleteSession(sessionId) {
    const response = await fetch(`/session/${sessionId}`, {
        method: 'DELETE',
    });
    if (!response.ok) {
        throw new Error('Delete failed');
    }
    return response.json();
}

// ============== Event Handlers ==============

async function handleLogin(e) {
    e.preventDefault();
    const password = elements.passwordInput.value;
    
    try {
        await login(password);
        elements.logoutBtn.style.display = 'block';
        showSection('uploadSection');
    } catch (error) {
        elements.loginError.textContent = error.message;
        elements.loginError.style.display = 'block';
    }
}

function handleFileSelect(file) {
    if (!file) return;
    
    if (!file.name.endsWith('.dcm')) {
        alert('Please select a .dcm file');
        return;
    }
    
    startUpload(file);
}

async function startUpload(file) {
    showSection('processingSection');
    setProgress(5, 'Uploading...');
    
    try {
        const result = await uploadFile(file);
        currentSessionId = result.session_id;
        elements.sessionIdDisplay.textContent = currentSessionId;
        
        // Start polling for progress
        pollProgress();
    } catch (error) {
        showError(error.message);
    }
}

function renderSimulation(simulation) {
    const el = document.getElementById('simulation-summary');
    if (!el) return;
    if (!simulation) {
        el.textContent = t('results_sim_none');
        return;
    }
    const s = simulation.scenario || {};
    const plan = [
        s.advance_mm ? `${s.advance_mm > 0 ? '+' : ''}${s.advance_mm} mm` : null,
        s.vertical_mm ? `${s.vertical_mm > 0 ? '+' : ''}${s.vertical_mm} mm vert` : null,
        s.lateral_mm ? `${s.lateral_mm > 0 ? '+' : ''}${s.lateral_mm} mm lat` : null,
        s.pitch_deg ? `${s.pitch_deg > 0 ? '+' : ''}${s.pitch_deg}°` : null,
    ].filter(Boolean).join(', ') || 'no movement';
    el.textContent = `${plan} → max ${simulation.max_disp_mm} mm, mean ${simulation.mean_disp_mm} mm`;
}

function pollProgress() {
    if (!currentSessionId) return;
    
    progressPollInterval = setInterval(async () => {
        try {
            const status = await getSessionStatus(currentSessionId);
            
            if (status.status === 'completed') {
                clearInterval(progressPollInterval);
                setProgress(100, 'Complete!');
                showSection('resultsSection');
                elements.downloadBtn.href = status.download_url;
                renderSimulation(status.simulation);
            } else if (status.status === 'failed') {
                clearInterval(progressPollInterval);
                showError(status.error || 'Processing failed');
            } else if (status.status === 'processing') {
                // Map status to progress (simplified - backend should send detailed progress)
                setProgress(50, 'Processing...');
            }
        } catch (error) {
            console.error('Polling error:', error);
        }
    }, 2000);
}

function showError(message) {
    elements.errorDetails.textContent = message;
    showSection('errorSection');
}

function resetApp() {
    currentSessionId = null;
    if (progressPollInterval) {
        clearInterval(progressPollInterval);
    }
    setProgress(0, '');
    showSection('uploadSection');
}

// ============== Initialization ==============

function init() {
    // Load saved preferences
    const savedLang = localStorage.getItem('language') || 'en';
    currentLang = savedLang;
    elements.languageSelector.value = savedLang;
    
    const savedTheme = localStorage.getItem('theme') === 'dark';
    setTheme(savedTheme);
    
    updateTranslations();
    
    // Event listeners
    elements.loginForm.addEventListener('submit', handleLogin);
    
    elements.uploadArea.addEventListener('click', () => {
        elements.fileInput.click();
    });
    
    elements.fileInput.addEventListener('change', (e) => {
        handleFileSelect(e.target.files[0]);
    });
    
    elements.uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.add('dragover');
    });
    
    elements.uploadArea.addEventListener('dragleave', () => {
        elements.uploadArea.classList.remove('dragover');
    });
    
    elements.uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        elements.uploadArea.classList.remove('dragover');
        handleFileSelect(e.dataTransfer.files[0]);
    });
    
    elements.downloadBtn.addEventListener('click', () => {
        // Download will happen automatically via href
    });
    
    elements.newScanBtn.addEventListener('click', resetApp);
    
    elements.deleteSessionBtn.addEventListener('click', async () => {
        if (currentSessionId && confirm('Delete this session?')) {
            try {
                await deleteSession(currentSessionId);
                resetApp();
            } catch (error) {
                alert('Failed to delete session: ' + error.message);
            }
        }
    });
    
    elements.retryBtn.addEventListener('click', resetApp);
    
    elements.logoutBtn.addEventListener('click', async () => {
        await logout();
        elements.logoutBtn.style.display = 'none';
        showSection('loginSection');
    });
    
    elements.languageSelector.addEventListener('change', (e) => {
        currentLang = e.target.value;
        localStorage.setItem('language', currentLang);
        updateTranslations();
    });
    
    elements.themeToggle.addEventListener('click', toggleTheme);
}

// Start app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
