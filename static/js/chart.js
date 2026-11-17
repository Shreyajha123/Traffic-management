/**
 * TrafficSense AI - Chart.js Visualizations
 * Provides dynamic analytics rendering for Dashboard and Prediction benchmarks.
 */

// Keep track of active chart instances to properly destroy before re-rendering
const chartInstances = {};

function destroyChartIfExists(chartId) {
    if (chartInstances[chartId]) {
        chartInstances[chartId].destroy();
        delete chartInstances[chartId];
    }
}

/**
 * Common Chart.js theme options for dark navy palette
 */
const darkChartDefaults = {
    color: '#94a3b8',
    borderColor: 'rgba(255, 255, 255, 0.08)',
    font: {
        family: "'Inter', sans-serif",
        size: 11
    }
};

/**
 * Initializes and populates all 4 dashboard charts and KPI cards from /api/dashboard
 */
async function initDashboardCharts() {
    try {
        const response = await fetch('/api/dashboard');
        const data = await response.json();

        if (!data.success) {
            console.error('Failed to load dashboard data:', data.error);
            return;
        }

        // 1. Update KPI Metric Cards
        const metrics = data.metrics;
        document.getElementById('kpi-total-records').textContent = Number(metrics.total_records).toLocaleString();
        document.getElementById('kpi-avg-volume').textContent = metrics.avg_vehicle_count + ' veh';
        document.getElementById('kpi-avg-speed').textContent = metrics.avg_speed + ' km/h';
        document.getElementById('kpi-congestion-pct').textContent = metrics.congestion_percentage + '%';
        document.getElementById('kpi-peak-hour').textContent = metrics.peak_traffic_hour;

        // 2. Chart 1: Hourly Traffic & Speed Profile
        renderHourlyChart(data.charts.hourly);

        // 3. Chart 2: Congestion Distribution (Doughnut)
        renderCongestionDoughnut(data.charts.congestion);

        // 4. Chart 3: Road Volume Distribution (Horizontal / Vertical Bar)
        renderRoadVolumeChart(data.charts.road);

        // 5. Chart 4: Weather Impact Chart (Grouped Bar)
        renderWeatherImpactChart(data.charts.weather);

    } catch (err) {
        console.error('Error fetching dashboard API:', err);
    }
}

/**
 * Chart 1: Hourly Traffic Volume (Line) + Speed (Dashed Line)
 */
function renderHourlyChart(hourlyData) {
    const canvas = document.getElementById('chartHourlyTraffic');
    if (!canvas) return;

    destroyChartIfExists('hourly');

    chartInstances['hourly'] = new Chart(canvas, {
        type: 'line',
        data: {
            labels: hourlyData.hours,
            datasets: [
                {
                    label: 'Avg Vehicle Count (veh/hr)',
                    data: hourlyData.volumes,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.15)',
                    borderWidth: 2.5,
                    fill: true,
                    tension: 0.35,
                    pointBackgroundColor: '#3b82f6',
                    pointRadius: 3,
                    pointHoverRadius: 6,
                    yAxisID: 'y'
                },
                {
                    label: 'Avg Speed (km/h)',
                    data: hourlyData.speeds,
                    borderColor: '#06b6d4',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    tension: 0.35,
                    pointBackgroundColor: '#06b6d4',
                    pointRadius: 2.5,
                    pointHoverRadius: 5,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    labels: { color: darkChartDefaults.color, font: darkChartDefaults.font }
                },
                tooltip: {
                    backgroundColor: '#0f172a',
                    borderColor: 'rgba(255, 255, 255, 0.15)',
                    borderWidth: 1,
                    titleColor: '#ffffff',
                    bodyColor: '#e2e8f0',
                    padding: 10
                }
            },
            scales: {
                x: {
                    grid: { color: darkChartDefaults.borderColor },
                    ticks: { color: darkChartDefaults.color, maxTicksLimit: 12 }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'Vehicle Count', color: darkChartDefaults.color },
                    grid: { color: darkChartDefaults.borderColor },
                    ticks: { color: darkChartDefaults.color }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'Speed (km/h)', color: darkChartDefaults.color },
                    grid: { drawOnChartArea: false },
                    ticks: { color: darkChartDefaults.color }
                }
            }
        }
    });
}

/**
 * Chart 2: Congestion Distribution (Doughnut)
 */
function renderCongestionDoughnut(congestionData) {
    const canvas = document.getElementById('chartCongestionDist');
    if (!canvas) return;

    destroyChartIfExists('congestion');

    chartInstances['congestion'] = new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: congestionData.labels,
            datasets: [{
                data: congestionData.counts,
                backgroundColor: [
                    '#10b981', // Low: Emerald
                    '#f59e0b', // Medium: Amber
                    '#ef4444'  // High: Red
                ],
                borderColor: '#1e293b',
                borderWidth: 3,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: darkChartDefaults.color, font: darkChartDefaults.font, padding: 14 }
                },
                tooltip: {
                    backgroundColor: '#0f172a',
                    borderColor: 'rgba(255, 255, 255, 0.15)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(ctx) {
                            const val = ctx.raw || 0;
                            const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                            const pct = ((val / total) * 100).toFixed(1);
                            return ` ${ctx.label}: ${val} records (${pct}%)`;
                        }
                    }
                }
            },
            cutout: '68%'
        }
    });
}

/**
 * Chart 3: Road Corridor Average Volumes (Bar)
 */
function renderRoadVolumeChart(roadData) {
    const canvas = document.getElementById('chartRoadVolume');
    if (!canvas) return;

    destroyChartIfExists('road');

    chartInstances['road'] = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: roadData.roads,
            datasets: [{
                label: 'Avg Vehicle Volume (veh/hr)',
                data: roadData.volumes,
                backgroundColor: 'rgba(59, 130, 246, 0.75)',
                borderColor: '#3b82f6',
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#0f172a',
                    borderColor: 'rgba(255, 255, 255, 0.15)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: darkChartDefaults.color, font: { size: 10 } }
                },
                y: {
                    grid: { color: darkChartDefaults.borderColor },
                    ticks: { color: darkChartDefaults.color },
                    title: { display: true, text: 'Avg Vehicles / Hour', color: darkChartDefaults.color }
                }
            }
        }
    });
}

/**
 * Chart 4: Weather vs Volume & Speed (Grouped Bar)
 */
function renderWeatherImpactChart(weatherData) {
    const canvas = document.getElementById('chartWeatherImpact');
    if (!canvas) return;

    destroyChartIfExists('weather');

    chartInstances['weather'] = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: weatherData.conditions,
            datasets: [
                {
                    label: 'Avg Volume (veh/hr)',
                    data: weatherData.volumes,
                    backgroundColor: 'rgba(139, 92, 246, 0.75)',
                    borderColor: '#8b5cf6',
                    borderWidth: 1.5,
                    borderRadius: 4,
                    yAxisID: 'y'
                },
                {
                    label: 'Avg Speed (km/h)',
                    data: weatherData.speeds,
                    backgroundColor: 'rgba(6, 182, 212, 0.75)',
                    borderColor: '#06b6d4',
                    borderWidth: 1.5,
                    borderRadius: 4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: darkChartDefaults.color, font: darkChartDefaults.font }
                },
                tooltip: {
                    backgroundColor: '#0f172a',
                    borderColor: 'rgba(255, 255, 255, 0.15)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: darkChartDefaults.color }
                },
                y: {
                    type: 'linear',
                    position: 'left',
                    grid: { color: darkChartDefaults.borderColor },
                    ticks: { color: darkChartDefaults.color },
                    title: { display: true, text: 'Vehicle Volume', color: darkChartDefaults.color }
                },
                y1: {
                    type: 'linear',
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    ticks: { color: darkChartDefaults.color },
                    title: { display: true, text: 'Speed (km/h)', color: darkChartDefaults.color }
                }
            }
        }
    });
}

/**
 * Predict Page: Benchmark Chart comparing Prediction vs Threshold Boundaries
 */
function renderPredictionBenchmarkChart(predictedVolume, roadName) {
    const canvas = document.getElementById('chartPredictionBenchmark');
    if (!canvas) return;

    destroyChartIfExists('benchmark');

    let barColor = '#10b981';
    if (predictedVolume > 700) {
        barColor = '#ef4444';
    } else if (predictedVolume >= 300) {
        barColor = '#f59e0b';
    }

    chartInstances['benchmark'] = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: ['Low Max (299)', 'Medium Max (700)', `${roadName} Prediction`],
            datasets: [{
                label: 'Vehicles / Hour',
                data: [300, 700, predictedVolume],
                backgroundColor: [
                    'rgba(16, 185, 129, 0.35)',
                    'rgba(245, 158, 11, 0.35)',
                    barColor
                ],
                borderColor: [
                    '#10b981',
                    '#f59e0b',
                    barColor
                ],
                borderWidth: 2,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#0f172a',
                    borderColor: 'rgba(255, 255, 255, 0.15)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { color: darkChartDefaults.borderColor },
                    ticks: { color: darkChartDefaults.color },
                    title: { display: true, text: 'Vehicles / Hour', color: darkChartDefaults.color }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: darkChartDefaults.color }
                }
            }
        }
    });
}
