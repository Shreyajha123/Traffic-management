/**
 * TrafficSense AI - Master Frontend JavaScript
 * Handles navigation, ML model status, Prediction inference, AI Agent, Chatbot, and Dataset table.
 */

// Global state storage
let currentDatasetPage = 1;
const datasetPageLimit = 20;
let lastPredictionData = null;
let searchDebounceTimeout = null;

document.addEventListener("DOMContentLoaded", function() {
    initSidebarToggle();
    checkSystemStatus();
});

/**
 * 1. Sidebar Toggle & Mobile Navigation
 */
function initSidebarToggle() {
    const menuToggle = document.getElementById("menuToggle");
    const sidebar = document.getElementById("sidebar");

    if (menuToggle && sidebar) {
        menuToggle.addEventListener("click", function(e) {
            e.stopPropagation();
            sidebar.classList.toggle("open");
        });

        // Close sidebar when clicking outside on mobile
        document.addEventListener("click", function(e) {
            if (window.innerWidth <= 768 && sidebar.classList.contains("open") && !sidebar.contains(e.target)) {
                sidebar.classList.remove("open");
            }
        });
    }
}

/**
 * 2. System Status Polling (/api/model-status)
 */
async function checkSystemStatus() {
    const indicator = document.getElementById("model-status-indicator");
    const statusText = document.getElementById("model-status-text");
    const geminiText = document.getElementById("gemini-status-text");

    try {
        const response = await fetch("/api/model-status");
        const data = await response.json();

        if (indicator && statusText) {
            if (data.model_trained) {
                indicator.className = "status-indicator online";
                statusText.textContent = "ML Model: Online";
            } else {
                indicator.className = "status-indicator offline";
                statusText.textContent = "ML Model: Offline";
            }
        }

        if (geminiText) {
            if (data.gemini_configured) {
                geminiText.textContent = "Gemini: Active (Live API)";
                geminiText.style.color = "#6ee7b7";
            } else {
                geminiText.textContent = "Gemini: Educational Mode";
                geminiText.style.color = "#94a3b8";
            }
        }

        // Update hero records if on home page
        const heroRecords = document.getElementById("hero-total-records");
        if (heroRecords && data.total_records) {
            heroRecords.textContent = Number(data.total_records).toLocaleString();
        }

        // Update assistant subtitle if on assistant page
        const assistantEngineSub = document.getElementById("assistantEngineSub");
        if (assistantEngineSub) {
            assistantEngineSub.textContent = data.gemini_configured
                ? "Powered by Google Gemini 2.5 Flash API"
                : "Powered by TrafficSense Educational Knowledge Engine";
        }

    } catch (err) {
        console.warn("Could not reach model status endpoint:", err);
        if (indicator && statusText) {
            indicator.className = "status-indicator offline";
            statusText.textContent = "System: Disconnected";
        }
    }
}

/**
 * 3. ML Prediction Submission
 */
async function handlePredictionSubmit(event) {
    event.preventDefault();

    const roadName = document.getElementById("road_name").value;
    const predDate = document.getElementById("pred_date").value;
    const predHour = parseInt(document.getElementById("pred_hour").value);
    const predWeather = document.getElementById("pred_weather").value;

    const emptyState = document.getElementById("predictionEmptyState");
    const errorState = document.getElementById("predictionErrorState");
    const errorMessage = document.getElementById("predictionErrorMessage");
    const resultCard = document.getElementById("predictionResultCard");
    const btnSubmit = document.getElementById("btnPredictSubmit");

    // Hide error
    errorState.style.display = "none";

    // Basic Validation
    if (!roadName || !predDate || isNaN(predHour) || !predWeather) {
        errorMessage.textContent = "Please fill in all input fields.";
        errorState.style.display = "flex";
        return;
    }

    // Button loading animation
    const originalBtnHtml = btnSubmit.innerHTML;
    btnSubmit.disabled = true;
    btnSubmit.innerHTML = `<span class="loading-spinner"></span> Computing Random Forest Inference...`;

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                road_name: roadName,
                date: predDate,
                hour: predHour,
                weather: predWeather
            })
        });

        const data = await response.json();

        if (!data.success) {
            emptyState.style.display = "none";
            resultCard.style.display = "none";
            errorMessage.textContent = data.error || "Prediction request failed.";
            errorState.style.display = "flex";
            return;
        }

        const pred = data.prediction;
        lastPredictionData = pred; // Save for AI Agent

        // Hide empty & error states, reveal result card
        emptyState.style.display = "none";
        errorState.style.display = "none";
        resultCard.style.display = "block";

        // 1. Animate Predicted Volume Number
        animateCounter(document.getElementById("resVolume"), 0, pred.predicted_count, 600);

        // 2. Congestion Badge
        const badgeBox = document.getElementById("resCongestionBadge");
        const badgeText = document.getElementById("resCongestionText");
        const badgeSub = document.getElementById("resCongestionSub");

        badgeText.textContent = `${pred.congestion_level} Congestion`;
        if (pred.congestion_level === "High") {
            badgeBox.className = "congestion-badge-box badge-high";
            badgeSub.textContent = "Above 700 vehicles/hr";
        } else if (pred.congestion_level === "Medium") {
            badgeBox.className = "congestion-badge-box badge-medium";
            badgeSub.textContent = "300 - 700 vehicles/hr";
        } else {
            badgeBox.className = "congestion-badge-box badge-low";
            badgeSub.textContent = "Below 300 vehicles/hr";
        }

        // 3. Estimated Speed
        document.getElementById("resSpeed").textContent = `${pred.estimated_speed}`;

        // 4. Threshold Meter Bar & Indicator
        // Normalize 0 to 1200 vehicles to 0-100%
        const maxScale = 1200;
        const pct = Math.min(100, Math.max(0, (pred.predicted_count / maxScale) * 100));
        document.getElementById("meterFill").style.width = `${pct}%`;
        document.getElementById("meterPointer").style.left = `${pct}%`;
        document.getElementById("pointerText").textContent = `${pred.predicted_count} veh`;

        // 5. Context Chips
        document.getElementById("chipRoad").textContent = pred.road_name;
        document.getElementById("chipDate").textContent = `${pred.date} (${pred.day_of_week})`;
        document.getElementById("chipTime").textContent = pred.time_display;
        document.getElementById("chipWeather").textContent = pred.weather;

        // 6. Benchmark Comparison Chart
        if (typeof renderPredictionBenchmarkChart === "function") {
            renderPredictionBenchmarkChart(pred.predicted_count, pred.road_name);
        }

        // 7. Update Agent prompt notification
        const agentPromptNotice = document.getElementById("agentInitialPrompt");
        if (agentPromptNotice) {
            agentPromptNotice.innerHTML = `
                <i class="fa-solid fa-circle-check" style="color: #10b981;"></i>
                <span>Prediction loaded for <strong>${pred.road_name} (${pred.congestion_level} Congestion - ${pred.predicted_count} veh/hr)</strong>. Click <strong>Generate Management Directives</strong> below to synthesize municipal policies.</span>
            `;
        }

    } catch (err) {
        emptyState.style.display = "none";
        resultCard.style.display = "none";
        errorMessage.textContent = "Network error: Unable to connect to prediction API.";
        errorState.style.display = "flex";
    } finally {
        btnSubmit.disabled = false;
        btnSubmit.innerHTML = originalBtnHtml;
    }
}

/**
 * Counter Animation Helper
 */
function animateCounter(elem, start, end, duration) {
    if (!elem) return;
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const current = Math.floor(progress * (end - start) + start);
        elem.textContent = current.toLocaleString();
        if (progress < 1) {
            window.requestAnimationFrame(step);
        } else {
            elem.textContent = end.toLocaleString();
        }
    };
    window.requestAnimationFrame(step);
}

/**
 * 4. Autonomous AI Traffic Management Agent
 */
async function triggerAIAgent() {
    const btn = document.getElementById("btnRunAgent");
    const initialPrompt = document.getElementById("agentInitialPrompt");
    const loadingState = document.getElementById("agentLoading");
    const outputContainer = document.getElementById("agentOutputContainer");
    const markdownOutput = document.getElementById("agentMarkdownOutput");
    const sourceTag = document.getElementById("agentSourceTag");

    if (!lastPredictionData) {
        alert("Please run a vehicle volume prediction above before launching the AI Management Agent.");
        return;
    }

    initialPrompt.style.display = "none";
    outputContainer.style.display = "none";
    loadingState.style.display = "flex";
    btn.disabled = true;

    try {
        const response = await fetch("/api/agent", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(lastPredictionData)
        });

        const data = await response.json();
        loadingState.style.display = "none";

        if (!data.success) {
            initialPrompt.style.display = "flex";
            alert("Agent failed to synthesize directives: " + (data.error || "Unknown error"));
            return;
        }

        outputContainer.style.display = "block";
        sourceTag.innerHTML = `<i class="fa-solid fa-microchip"></i> Source: ${data.source}`;

        // Render Markdown
        if (typeof marked !== "undefined" && marked.parse) {
            markdownOutput.innerHTML = marked.parse(data.suggestions);
        } else {
            markdownOutput.innerHTML = `<pre style="white-space: pre-wrap; font-family: inherit;">${data.suggestions}</pre>`;
        }

    } catch (err) {
        loadingState.style.display = "none";
        initialPrompt.style.display = "flex";
        alert("Network error communicating with AI Agent API.");
    } finally {
        btn.disabled = false;
    }
}

/**
 * 5. AI Assistant Chatbot
 */
async function handleChatSubmit(event) {
    event.preventDefault();

    const input = document.getElementById("chatInput");
    const message = input.value.trim();
    if (!message) return;

    input.value = "";
    appendChatMessage("user", message);

    // Bot Thinking Bubble
    const tempId = "bot-loading-" + Date.now();
    appendLoadingBubble(tempId);

    try {
        const response = await fetch("/api/assistant", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();
        removeBubble(tempId);

        if (data.success) {
            appendChatMessage("bot", data.reply, data.source, data.is_demo);
        } else {
            appendChatMessage("bot", "Sorry, an error occurred while generating a response. " + (data.error || ""));
        }
    } catch (err) {
        removeBubble(tempId);
        appendChatMessage("bot", "Network error: Unable to connect to the Assistant API endpoint.");
    }
}

function appendChatMessage(sender, text, source = null, isDemo = false) {
    const container = document.getElementById("chatMessages");
    if (!container) return;

    const bubble = document.createElement("div");
    bubble.className = sender === "user" ? "chat-bubble user-bubble" : "chat-bubble bot-bubble";

    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    let bodyHtml = "";
    if (sender === "user") {
        bodyHtml = `
            <div class="bubble-avatar"><i class="fa-solid fa-user"></i></div>
            <div class="bubble-body">
                <div class="bubble-author">You</div>
                <div class="bubble-text">${escapeHtml(text)}</div>
                <div class="bubble-time">${now}</div>
            </div>
        `;
    } else {
        let formattedText = text;
        if (typeof marked !== "undefined" && marked.parse) {
            formattedText = marked.parse(text);
        }

        let sourceLabel = "";
        if (source) {
            const badgeClass = isDemo ? "badge-demo" : "badge-live";
            sourceLabel = `<div style="font-size: 0.72rem; color: #94a3b8; margin-top: 6px; font-style: italic;"><i class="fa-solid fa-sparkles"></i> ${source}</div>`;
        }

        bodyHtml = `
            <div class="bubble-avatar"><i class="fa-solid fa-robot"></i></div>
            <div class="bubble-body">
                <div class="bubble-author">TrafficSense AI Assistant</div>
                <div class="bubble-text">${formattedText} ${sourceLabel}</div>
                <div class="bubble-time">${now}</div>
            </div>
        `;
    }

    bubble.innerHTML = bodyHtml;
    container.appendChild(bubble);
    container.scrollTop = container.scrollHeight;
}

function appendLoadingBubble(tempId) {
    const container = document.getElementById("chatMessages");
    if (!container) return;

    const bubble = document.createElement("div");
    bubble.className = "chat-bubble bot-bubble";
    bubble.id = tempId;
    bubble.innerHTML = `
        <div class="bubble-avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="bubble-body">
            <div class="bubble-author">TrafficSense AI Assistant</div>
            <div class="bubble-text" style="display: flex; align-items: center; gap: 8px;">
                <span class="loading-spinner"></span> Analyzing Bengaluru mobility dynamics...
            </div>
        </div>
    `;
    container.appendChild(bubble);
    container.scrollTop = container.scrollHeight;
}

function removeBubble(id) {
    const elem = document.getElementById(id);
    if (elem) elem.remove();
}

function clearChatHistory() {
    const container = document.getElementById("chatMessages");
    if (container) {
        container.innerHTML = `
            <div class="chat-bubble bot-bubble">
                <div class="bubble-avatar"><i class="fa-solid fa-robot"></i></div>
                <div class="bubble-body">
                    <div class="bubble-author">TrafficSense AI Assistant</div>
                    <div class="bubble-text">
                        <p>Chat history cleared. How can I assist you with urban traffic analysis today?</p>
                    </div>
                    <div class="bubble-time">System Ready</div>
                </div>
            </div>
        `;
    }
}

/**
 * 6. Dataset Explorer & Report Table
 */
async function loadDatasetTable() {
    const tbody = document.getElementById("dataTableBody");
    if (!tbody) return;

    const road = document.getElementById("filterRoad")?.value || "all";
    const weather = document.getElementById("filterWeather")?.value || "all";
    const congestion = document.getElementById("filterCongestion")?.value || "all";
    const search = document.getElementById("dataSearchInput")?.value || "";

    const queryParams = new URLSearchParams({
        page: currentDatasetPage,
        limit: datasetPageLimit,
        road: road,
        weather: weather,
        congestion: congestion,
        search: search
    });

    tbody.innerHTML = `
        <tr>
            <td colspan="8" class="text-center py-4" style="text-align: center; padding: 24px;">
                <div class="loading-spinner"></div> Loading dataset records...
            </td>
        </tr>
    `;

    try {
        const response = await fetch(`/api/dataset?${queryParams.toString()}`);
        const result = await response.json();

        if (!result.success || !result.data || result.data.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="8" style="text-align: center; padding: 24px; color: #94a3b8;">
                        No records found matching the specified filters.
                    </td>
                </tr>
            `;
            updatePaginationUI(0, 0, 0);
            return;
        }

        tbody.innerHTML = result.data.map(row => {
            let badgeClass = "badge-low";
            if (row.congestion_level === "High") badgeClass = "badge-high";
            else if (row.congestion_level === "Medium") badgeClass = "badge-medium";

            return `
                <tr>
                    <td><strong>${row.date}</strong></td>
                    <td>${String(row.hour).padStart(2, '0')}:00</td>
                    <td>${row.day_of_week}</td>
                    <td><span style="color: #ffffff; font-weight: 600;">${row.road_name}</span></td>
                    <td>${row.weather}</td>
                    <td><span style="font-weight: 700; color: #60a5fa;">${row.vehicle_count}</span></td>
                    <td>${row.avg_speed} km/h</td>
                    <td>
                        <span class="congestion-badge-box ${badgeClass}" style="font-size: 0.72rem; padding: 3px 8px;">
                            <span class="badge-dot"></span> ${row.congestion_level}
                        </span>
                    </td>
                </tr>
            `;
        }).join("");

        updatePaginationUI(result.total, result.page, result.total_pages);

        const totalCountElem = document.getElementById("dataTotalCount");
        if (totalCountElem) {
            totalCountElem.textContent = Number(result.total).toLocaleString();
        }

    } catch (err) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" style="text-align: center; padding: 24px; color: #ef4444;">
                    Error loading dataset records. Please check backend server.
                </td>
            </tr>
        `;
    }
}

function handleDataFilterChange() {
    clearTimeout(searchDebounceTimeout);
    searchDebounceTimeout = setTimeout(() => {
        currentDatasetPage = 1;
        loadDatasetTable();
    }, 250);
}

function resetDataFilters() {
    document.getElementById("filterRoad").value = "all";
    document.getElementById("filterWeather").value = "all";
    document.getElementById("filterCongestion").value = "all";
    document.getElementById("dataSearchInput").value = "";
    currentDatasetPage = 1;
    loadDatasetTable();
}

function changeDataPage(delta) {
    currentDatasetPage += delta;
    loadDatasetTable();
}

function updatePaginationUI(total, page, totalPages) {
    const pageInfo = document.getElementById("tablePageInfo");
    const pageIndicator = document.getElementById("pageIndicator");
    const btnPrev = document.getElementById("btnPrevPage");
    const btnNext = document.getElementById("btnNextPage");

    if (total === 0) {
        if (pageInfo) pageInfo.textContent = "Showing 0 records";
        if (pageIndicator) pageIndicator.textContent = "Page 0 of 0";
        if (btnPrev) btnPrev.disabled = true;
        if (btnNext) btnNext.disabled = true;
        return;
    }

    const start = (page - 1) * datasetPageLimit + 1;
    const end = Math.min(start + datasetPageLimit - 1, total);

    if (pageInfo) pageInfo.textContent = `Showing records ${start} - ${end} of ${total.toLocaleString()}`;
    if (pageIndicator) pageIndicator.textContent = `Page ${page} of ${totalPages}`;
    if (btnPrev) btnPrev.disabled = (page <= 1);
    if (btnNext) btnNext.disabled = (page >= totalPages);
}

/**
 * Utility: HTML Escape
 */
function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}
