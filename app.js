let stockChart = null; 
let isPredictionPageActive = false;

function generateSimulatedData(ticker) {
    const days = 150;
    const dates = [];
    const truePrices = [];
    const predictedPrices = [];
    let mse = 0;
    let lastPrice = 100;
    const today = new Date();

    for (let i = days - 1; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(today.getDate() - i);
        dates.push(date.toISOString().split('T')[0]);
        
        lastPrice += (Math.random() - 0.5) * 2.5;
        lastPrice = Math.max(50, lastPrice); 
        truePrices.push(parseFloat(lastPrice.toFixed(2)));
        
        const pred = lastPrice * (1 + (Math.random() - 0.5) * 0.05);
        predictedPrices.push(parseFloat(pred.toFixed(2)));

        mse += Math.pow(truePrices[truePrices.length - 1] - predictedPrices[predictedPrices.length - 1], 2);
    }
    
    return {
        ticker: ticker,
        dates: dates,
        truePrices: truePrices,
        predictedPrices: predictedPrices,
        mse: mse / days
    };
}

function renderChart(data) {
    const chartCanvas = document.getElementById('stockChart');
    if (!chartCanvas) return; 
    
    const ctx = chartCanvas.getContext('2d');
    
    if (stockChart) {
        stockChart.destroy();
    }

    stockChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: 'True Mid Price',
                    data: data.truePrices,
                    borderColor: '#00aaff', 
                    backgroundColor: 'rgba(0, 170, 255, 0.1)',
                    borderWidth: 3,
                    tension: 0.1,
                    pointRadius: 0
                },
                {
                    label: 'Predicted Mid Price',
                    data: data.predictedPrices,
                    borderColor: '#ff5555', 
                    backgroundColor: 'transparent',
                    borderWidth: 3,
                    tension: 0.1,
                    pointRadius: 0,
                    borderDash: [8, 4] 
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `${data.ticker} Stock Price Prediction (USD)`,
                    font: { size: 18 },
                    color: '#ffffff' 
                },
                legend: {
                    position: 'top',
                    labels: {
                        color: '#ffffff' 
                    }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(18, 18, 18, 0.8)', 
                    titleColor: '#ffffff',
                    bodyColor: '#cccccc'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date',
                        color: '#cccccc' 
                    },
                    ticks: {
                        color: '#cccccc' 
                    },
                    grid: {
                        color: '#333333', 
                        drawBorder: false
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Mid Price (USD)',
                        color: '#cccccc'
                    },
                    ticks: {
                        color: '#cccccc'
                    },
                    grid: {
                        color: '#333333', 
                        drawBorder: false
                    }
                }
            }
        }
    });
}

function setupPredictionPage() {
    if (!isPredictionPageActive) return;

    const form = document.getElementById('prediction-form');
    if (form && !form.dataset.listenerAttached) {
        form.dataset.listenerAttached = 'true';
        
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const loadingSpinner = document.getElementById('loading-spinner');
            const statusMessage = document.getElementById('status-message');
            const mseResult = document.getElementById('mse-result');
            
            const ticker = document.getElementById('ticker').value.toUpperCase();
            const epochs = parseInt(document.getElementById('epochs').value);
            const lookback = parseInt(document.getElementById('lookback').value);

            if (!ticker || isNaN(epochs) || isNaN(lookback)) {
                statusMessage.textContent = "Validation error: Please fill in all valid fields.";
                console.error("Please fill in all valid fields.");
                return;
            }

            loadingSpinner.classList.remove('d-none');
            statusMessage.textContent = `Requesting prediction for ${ticker} with ${epochs} epochs...`;
            mseResult.textContent = '--';
            
            setTimeout(() => {
                const simulatedData = generateSimulatedData(ticker);
                renderChart(simulatedData);
                loadingSpinner.classList.add('d-none');
                statusMessage.textContent = `Prediction complete for ${ticker}. (Simulated Data)`;
                mseResult.textContent = simulatedData.mse.toFixed(6);
            }, 3000); 
        });
    }
}


function renderPage(pageId) {
    const mainContent = document.getElementById('main-content');
    const template = document.getElementById(`${pageId}-template`);
    
    if (!template) {
        mainContent.innerHTML = `<h1 class="text-danger">Error: Page template for ${pageId} not found.</h1>`;
        isPredictionPageActive = false;
        return;
    }

    mainContent.innerHTML = template.innerHTML;

    document.querySelectorAll('.sidebar-link').forEach(link => {
        link.classList.remove('active');
        if (link.dataset.page === pageId) {
            link.classList.add('active');
        }
    });

    isPredictionPageActive = (pageId === 'prediction');
    if (isPredictionPageActive) {
        setTimeout(setupPredictionPage, 0); 
    } else {
        if (stockChart) {
            stockChart.destroy();
            stockChart = null;
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.sidebar-link').forEach(link => {
        link.addEventListener('click', function() {
            const pageId = this.dataset.page;
            renderPage(pageId);
        });
    });

    renderPage('home');
});