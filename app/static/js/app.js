// Sentiment Analysis App JavaScript
class SentimentApp {
    constructor() {
        this.charts = {};
        this.history = [];
        this.isLoading = false;
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadModelInfo();
        this.updateDashboard();
        this.startAutoRefresh();
    }
    
    setupEventListeners() {
        // Single text analysis
        document.getElementById('sentiment-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyzeSingleText();
        });
        
        // Character counter
        document.getElementById('text-input').addEventListener('input', (e) => {
            this.updateCharCounter(e.target.value);
        });
        
        // Clear button
        document.getElementById('clear-btn').addEventListener('click', () => {
            this.clearSingleTextForm();
        });
        
        // Example buttons
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const text = e.target.getAttribute('data-text');
                document.getElementById('text-input').value = text;
                this.updateCharCounter(text);
            });
        });
        
        // Batch analysis
        document.getElementById('batch-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.analyzeBatch();
        });
        
        // Line counter for batch
        document.getElementById('batch-input').addEventListener('input', (e) => {
            this.updateLineCounter(e.target.value);
        });
        
        // Compare functionality
        document.getElementById('compare-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.compareTexts();
        });
        
        // Add compare text button
        document.getElementById('add-compare-btn').addEventListener('click', () => {
            this.addCompareInput();
        });
        
        // Remove compare text buttons (delegated)
        document.getElementById('compare-inputs').addEventListener('click', (e) => {
            if (e.target.closest('.remove-compare-btn')) {
                this.removeCompareInput(e.target.closest('.compare-input-row'));
            }
        });
        
        // Clear history button
        document.getElementById('clear-history-btn').addEventListener('click', () => {
            this.clearHistory();
        });
        
        // Tab change events
        document.querySelectorAll('[data-bs-toggle="tab"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                const target = e.target.getAttribute('href').substring(1);
                this.onTabChange(target);
            });
        });
    }
    
    // ============ SINGLE TEXT ANALYSIS ============
    
    async analyzeSingleText() {
        const textInput = document.getElementById('text-input');
        const text = textInput.value.trim();
        
        if (!text) {
            this.showToast('Please enter some text to analyze', 'warning');
            return;
        }
        
        this.setLoading(true);
        this.showLoadingCard();
        
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text, detailed: true })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displaySingleResult(result);
                this.showToast('Analysis completed successfully!', 'success');
            } else {
                throw new Error(result.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('Error analyzing text:', error);
            this.showToast(`Error: ${error.message}`, 'error');
            this.hideResultCard();
        } finally {
            this.setLoading(false);
            this.hideLoadingCard();
        }
    }
    
    displaySingleResult(result) {
        const resultCard = document.getElementById('result-card');
        
        // Update star rating
        this.updateStarRating(result.sentiment_score);
        
        // Update score display
        document.getElementById('sentiment-score').textContent = `${result.sentiment_score}/5.0`;
        document.getElementById('sentiment-class').textContent = 
            result.prediction_class.charAt(0).toUpperCase() + result.prediction_class.slice(1);
        document.getElementById('sentiment-class').className = `text-muted sentiment-${result.prediction_class}`;
        
        // Update confidence bar
        const confidencePercent = Math.round(result.confidence * 100);
        const confidenceBar = document.getElementById('confidence-bar');
        confidenceBar.style.width = `${confidencePercent}%`;
        confidenceBar.className = `progress-bar ${this.getConfidenceClass(result.confidence)}`;
        document.getElementById('confidence-text').textContent = `${confidencePercent}%`;
        
        // Update interpretation
        const interpretationDiv = document.getElementById('interpretation');
        if (result.sentiment_analysis && result.sentiment_analysis.interpretation) {
            interpretationDiv.textContent = result.sentiment_analysis.interpretation;
            interpretationDiv.style.display = 'block';
        } else {
            interpretationDiv.style.display = 'none';
        }
        
        // Update processing time
        const processingTime = Math.round((result.processing_time || 0) * 1000);
        document.getElementById('processing-time').textContent = `Processing time: ${processingTime}ms`;
        
        // Show result card with animation
        resultCard.style.display = 'block';
        resultCard.classList.add('slide-in-up');
        
        setTimeout(() => {
            resultCard.classList.remove('slide-in-up');
        }, 500);
    }
    
    updateStarRating(score) {
        const starRating = document.getElementById('star-rating');
        const fullStars = Math.floor(score);
        const hasHalfStar = (score - fullStars) >= 0.5;
        const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);
        
        let starsHtml = '';
        
        // Full stars
        for (let i = 0; i < fullStars; i++) {
            starsHtml += '<span class="star full"><i class="fas fa-star"></i></span>';
        }
        
        // Half star
        if (hasHalfStar) {
            starsHtml += '<span class="star half"><i class="fas fa-star-half-alt"></i></span>';
        }
        
        // Empty stars
        for (let i = 0; i < emptyStars; i++) {
            starsHtml += '<span class="star empty"><i class="far fa-star"></i></span>';
        }
        
        starRating.innerHTML = starsHtml;
    }
    
    getConfidenceClass(confidence) {
        if (confidence >= 0.8) return 'confidence-very-high';
        if (confidence >= 0.6) return 'confidence-high';
        if (confidence >= 0.4) return 'confidence-moderate';
        return 'confidence-low';
    }
    
    // ============ BATCH ANALYSIS ============
    
    async analyzeBatch() {
        const batchInput = document.getElementById('batch-input');
        const text = batchInput.value.trim();
        
        if (!text) {
            this.showToast('Please enter texts to analyze (one per line)', 'warning');
            return;
        }
        
        const texts = text.split('\n').filter(line => line.trim()).slice(0, 100);
        
        if (texts.length === 0) {
            this.showToast('No valid texts found', 'warning');
            return;
        }
        
        this.setLoading(true);
        const analyzeBtn = document.getElementById('batch-analyze-btn');
        const originalText = analyzeBtn.innerHTML;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
        analyzeBtn.disabled = true;
        
        try {
            const response = await fetch('/api/batch_predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ texts: texts })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayBatchResults(result, texts);
                this.showToast(`Analyzed ${texts.length} texts successfully!`, 'success');
            } else {
                throw new Error(result.error || 'Batch analysis failed');
            }
        } catch (error) {
            console.error('Error in batch analysis:', error);
            this.showToast(`Error: ${error.message}`, 'error');
        } finally {
            this.setLoading(false);
            analyzeBtn.innerHTML = originalText;
            analyzeBtn.disabled = false;
        }
    }
    
    displayBatchResults(result, texts) {
        const { results, batch_statistics } = result;
        
        // Update summary card
        document.getElementById('positive-count').textContent = batch_statistics.class_distribution.positive;
        document.getElementById('neutral-count').textContent = batch_statistics.class_distribution.neutral;
        document.getElementById('negative-count').textContent = batch_statistics.class_distribution.negative;
        document.getElementById('avg-score').textContent = `${batch_statistics.mean_score}/5.0`;
        
        // Show summary card
        document.getElementById('batch-summary-card').style.display = 'block';
        
        // Update results table
        const tableBody = document.querySelector('#batch-results-table tbody');
        tableBody.innerHTML = '';
        
        results.forEach((result, index) => {
            const row = document.createElement('tr');
            row.className = 'batch-result-row';
            
            const truncatedText = texts[index].length > 50 ? 
                texts[index].substring(0, 50) + '...' : texts[index];
            
            const starsHtml = this.generateStarsHtml(result.sentiment_score);
            
            row.innerHTML = `
                <td>${index + 1}</td>
                <td title="${texts[index]}">${truncatedText}</td>
                <td>${result.sentiment_score}</td>
                <td><span class="badge sentiment-${result.prediction_class}">${result.prediction_class}</span></td>
                <td>${Math.round(result.confidence * 100)}%</td>
                <td>${starsHtml}</td>
            `;
            
            tableBody.appendChild(row);
        });
        
        // Show results table
        document.getElementById('batch-results-card').style.display = 'block';
    }
    
    generateStarsHtml(score) {
        const fullStars = Math.floor(score);
        const hasHalfStar = (score - fullStars) >= 0.5;
        const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);
        
        let html = '';
        
        for (let i = 0; i < fullStars; i++) {
            html += '<i class="fas fa-star text-warning"></i>';
        }
        
        if (hasHalfStar) {
            html += '<i class="fas fa-star-half-alt text-warning"></i>';
        }
        
        for (let i = 0; i < emptyStars; i++) {
            html += '<i class="far fa-star text-warning"></i>';
        }
        
        return html;
    }
    
    // ============ COMPARE FUNCTIONALITY ============
    
    async compareTexts() {
        const compareInputs = document.querySelectorAll('.compare-input-row');
        const texts = [];
        const labels = [];
        
        compareInputs.forEach(row => {
            const text = row.querySelector('.compare-text').value.trim();
            const label = row.querySelector('.compare-label').value.trim();
            
            if (text) {
                texts.push(text);
                labels.push(label || `Text ${texts.length}`);
            }
        });
        
        if (texts.length < 2) {
            this.showToast('Please enter at least 2 texts to compare', 'warning');
            return;
        }
        
        this.setLoading(true);
        
        try {
            const response = await fetch('/api/compare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ texts: texts, labels: labels })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayCompareResults(result);
                this.showToast('Comparison completed successfully!', 'success');
            } else {
                throw new Error(result.error || 'Comparison failed');
            }
        } catch (error) {
            console.error('Error comparing texts:', error);
            this.showToast(`Error: ${error.message}`, 'error');
        } finally {
            this.setLoading(false);
        }
    }
    
    displayCompareResults(result) {
        const { results, comparison_statistics } = result;
        const compareCards = document.getElementById('compare-cards');
        compareCards.innerHTML = '';
        
        // Create comparison cards
        results.forEach((item, index) => {
            const card = document.createElement('div');
            card.className = `col-md-6 col-lg-4 mb-3`;
            
            const prediction = item.prediction;
            const starsHtml = this.generateStarsHtml(prediction.sentiment_score);
            
            card.innerHTML = `
                <div class="card compare-card ${prediction.prediction_class}">
                    <div class="card-body">
                        <h6 class="card-title">${item.label}</h6>
                        <p class="card-text small">${item.text.substring(0, 100)}${item.text.length > 100 ? '...' : ''}</p>
                        <div class="text-center">
                            <div class="mb-2">${starsHtml}</div>
                            <h5>${prediction.sentiment_score}/5.0</h5>
                            <span class="badge sentiment-${prediction.prediction_class}">${prediction.prediction_class}</span>
                            <div class="small text-muted mt-1">Confidence: ${Math.round(prediction.confidence * 100)}%</div>
                        </div>
                    </div>
                </div>
            `;
            
            compareCards.appendChild(card);
        });
        
        // Create comparison chart
        this.createCompareChart(results);
        
        // Show results
        document.getElementById('compare-results').style.display = 'block';
    }
    
    createCompareChart(results) {
        const ctx = document.getElementById('compare-chart').getContext('2d');
        
        if (this.charts.compare) {
            this.charts.compare.destroy();
        }
        
        const labels = results.map(r => r.label);
        const scores = results.map(r => r.prediction.sentiment_score);
        const confidences = results.map(r => r.prediction.confidence * 100);
        
        this.charts.compare = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Sentiment Score',
                        data: scores,
                        backgroundColor: 'rgba(54, 162, 235, 0.8)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Confidence (%)',
                        data: confidences,
                        backgroundColor: 'rgba(255, 99, 132, 0.8)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Text Comparison Results'
                    }
                },
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        min: 0,
                        max: 5,
                        title: {
                            display: true,
                            text: 'Sentiment Score'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        min: 0,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Confidence (%)'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                }
            }
        });
    }
    
    addCompareInput() {
        const compareInputs = document.getElementById('compare-inputs');
        const currentCount = compareInputs.children.length;
        
        if (currentCount >= 10) {
            this.showToast('Maximum 10 texts allowed for comparison', 'warning');
            return;
        }
        
        const newRow = document.createElement('div');
        newRow.className = 'row mb-3 compare-input-row';
        newRow.innerHTML = `
            <div class="col-md-3">
                <input type="text" class="form-control compare-label" placeholder="Label ${currentCount + 1}" value="Text ${currentCount + 1}">
            </div>
            <div class="col-md-8">
                <textarea class="form-control compare-text" rows="2" placeholder="Enter text to compare..."></textarea>
            </div>
            <div class="col-md-1">
                <button type="button" class="btn btn-outline-danger remove-compare-btn">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
        
        compareInputs.appendChild(newRow);
        this.updateRemoveButtons();
    }
    
    removeCompareInput(row) {
        const compareInputs = document.getElementById('compare-inputs');
        
        if (compareInputs.children.length > 2) {
            row.remove();
            this.updateRemoveButtons();
        } else {
            this.showToast('At least 2 texts are required for comparison', 'warning');
        }
    }
    
    updateRemoveButtons() {
        const compareInputs = document.getElementById('compare-inputs');
        const removeButtons = compareInputs.querySelectorAll('.remove-compare-btn');
        
        removeButtons.forEach((btn, index) => {
            btn.disabled = compareInputs.children.length <= 2;
        });
    }
    
    // ============ DASHBOARD ============
    
    async updateDashboard() {
        try {
            const [historyResponse, analyticsResponse] = await Promise.all([
                fetch('/api/history'),
                fetch('/api/analytics')
            ]);
            
            if (historyResponse.ok) {
                const historyData = await historyResponse.json();
                this.updateHistoryTable(historyData.history);
            }
            
            if (analyticsResponse.ok) {
                const analyticsData = await analyticsResponse.json();
                if (analyticsData.analytics) {
                    this.updateDashboardCards(analyticsData.analytics);
                    this.createDashboardCharts(analyticsData.analytics);
                    this.updateInsights(analyticsData.analytics.insights || []);
                }
            }
        } catch (error) {
            console.error('Error updating dashboard:', error);
        }
    }
    
    updateDashboardCards(analytics) {
        document.getElementById('total-analyses').textContent = analytics.total_analyses;
        document.getElementById('avg-sentiment').textContent = analytics.score_statistics.mean || '0.0';
        document.getElementById('avg-confidence').textContent = 
            Math.round((analytics.confidence_statistics.mean || 0) * 100) + '%';
        
        // Determine dominant class
        const classCounts = analytics.class_distribution;
        const dominantClass = Object.keys(classCounts).reduce((a, b) => 
            classCounts[a] > classCounts[b] ? a : b
        );
        document.getElementById('dominant-class').textContent = 
            dominantClass.charAt(0).toUpperCase() + dominantClass.slice(1);
    }
    
    createDashboardCharts(analytics) {
        this.createSentimentDistributionChart(analytics.class_distribution);
        this.createHourlyActivityChart(analytics.hourly_analysis);
    }
    
    createSentimentDistributionChart(classDistribution) {
        const ctx = document.getElementById('sentiment-distribution-chart').getContext('2d');
        
        if (this.charts.sentimentDistribution) {
            this.charts.sentimentDistribution.destroy();
        }
        
        this.charts.sentimentDistribution = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: [
                        classDistribution.positive || 0,
                        classDistribution.neutral || 0,
                        classDistribution.negative || 0
                    ],
                    backgroundColor: [
                        '#28a745',
                        '#ffc107',
                        '#dc3545'
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }
    
    createHourlyActivityChart(hourlyAnalysis) {
        const ctx = document.getElementById('hourly-activity-chart').getContext('2d');
        
        if (this.charts.hourlyActivity) {
            this.charts.hourlyActivity.destroy();
        }
        
        const hours = Array.from({length: 24}, (_, i) => i);
        const counts = hours.map(h => hourlyAnalysis.counts[h] || 0);
        
        this.charts.hourlyActivity = new Chart(ctx, {
            type: 'line',
            data: {
                labels: hours.map(h => `${h}:00`),
                datasets: [{
                    label: 'Analyses',
                    data: counts,
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });
    }
    
    updateHistoryTable(history) {
        const tableBody = document.querySelector('#history-table tbody');
        tableBody.innerHTML = '';
        
        history.slice(0, 20).forEach(entry => {
            const row = document.createElement('tr');
            const time = new Date(entry.timestamp).toLocaleTimeString();
            
            row.innerHTML = `
                <td>${time}</td>
                <td title="${entry.text}">${entry.text}</td>
                <td>${entry.sentiment_score}</td>
                <td><span class="badge sentiment-${entry.prediction_class}">${entry.prediction_class}</span></td>
            `;
            
            tableBody.appendChild(row);
        });
    }
    
    updateInsights(insights) {
        const insightsList = document.getElementById('insights-list');
        
        if (insights.length === 0) {
            insightsList.innerHTML = '<p class="text-muted">No insights available yet. Start analyzing some text!</p>';
            return;
        }
        
        insightsList.innerHTML = insights.map(insight => 
            `<div class="insight-item">${insight}</div>`
        ).join('');
    }
    
    async clearHistory() {
        if (!confirm('Are you sure you want to clear all analysis history?')) {
            return;
        }
        
        try {
            const response = await fetch('/api/clear_history', {
                method: 'POST'
            });
            
            if (response.ok) {
                this.showToast('History cleared successfully', 'success');
                this.updateDashboard();
            } else {
                const result = await response.json();
                throw new Error(result.error || 'Failed to clear history');
            }
        } catch (error) {
            console.error('Error clearing history:', error);
            this.showToast(`Error: ${error.message}`, 'error');
        }
    }
    
    // ============ MODEL INFO ============
    
    async loadModelInfo() {
        try {
            const response = await fetch('/api/model_info');
            const result = await response.json();
            
            if (response.ok) {
                this.displayModelInfo(result.model_info, result.performance_stats);
            } else {
                console.error('Failed to load model info:', result.error);
            }
        } catch (error) {
            console.error('Error loading model info:', error);
        }
    }
    
    displayModelInfo(modelInfo, performanceStats) {
        // Model information table
        const modelInfoTable = document.getElementById('model-info-table');
        modelInfoTable.innerHTML = `
            <tr><th>Model Name</th><td>${modelInfo.model_name}</td></tr>
            <tr><th>Max Length</th><td>${modelInfo.max_length}</td></tr>
            <tr><th>Device</th><td>${modelInfo.device}</td></tr>
            <tr><th>Parameters</th><td>${modelInfo.num_parameters.toLocaleString()}</td></tr>
            <tr><th>Training Samples</th><td>${modelInfo.training_info.training_samples.toLocaleString()}</td></tr>
            <tr><th>Validation Samples</th><td>${modelInfo.training_info.validation_samples.toLocaleString()}</td></tr>
            <tr><th>Test Samples</th><td>${modelInfo.training_info.test_samples.toLocaleString()}</td></tr>
        `;
        
        // Performance metrics table
        const metricsTable = document.getElementById('performance-metrics-table');
        const metrics = modelInfo.performance_metrics;
        metricsTable.innerHTML = `
            <tr><th>Accuracy</th><td>${(metrics.accuracy * 100).toFixed(2)}%</td></tr>
            <tr><th>MAE</th><td>${metrics.mae.toFixed(4)}</td></tr>
            <tr><th>RMSE</th><td>${metrics.rmse.toFixed(4)}</td></tr>
            <tr><th>R² Score</th><td>${metrics.r2.toFixed(4)}</td></tr>
            <tr><th>Loss</th><td>${metrics.loss.toFixed(6)}</td></tr>
        `;
        
        // Runtime statistics
        document.getElementById('total-inferences').textContent = performanceStats.total_inferences;
        document.getElementById('avg-inference-time').textContent = 
            Math.round(performanceStats.average_inference_time * 1000) + 'ms';
        document.getElementById('cache-hit-rate').textContent = 
            Math.round(performanceStats.cache_hit_rate * 100) + '%';
        document.getElementById('cache-size').textContent = performanceStats.cache_size;
    }
    
    // ============ UTILITY FUNCTIONS ============
    
    updateCharCounter(text) {
        const counter = document.getElementById('char-count');
        const length = text.length;
        counter.textContent = length;
        
        if (length > 5000) {
            counter.className = 'char-counter danger';
        } else if (length > 3000) {
            counter.className = 'char-counter warning';
        } else {
            counter.className = 'char-counter';
        }
    }
    
    updateLineCounter(text) {
        const counter = document.getElementById('line-count');
        const lines = text.split('\n').filter(line => line.trim()).length;
        counter.textContent = lines;
        
        if (lines > 100) {
            counter.className = 'char-counter danger';
        } else if (lines > 75) {
            counter.className = 'char-counter warning';
        } else {
            counter.className = 'char-counter';
        }
    }
    
    clearSingleTextForm() {
        document.getElementById('text-input').value = '';
        this.updateCharCounter('');
        this.hideResultCard();
    }
    
    setLoading(loading) {
        this.isLoading = loading;
        document.body.style.cursor = loading ? 'wait' : 'default';
    }
    
    showLoadingCard() {
        document.getElementById('loading-card').style.display = 'block';
        document.getElementById('result-card').style.display = 'none';
    }
    
    hideLoadingCard() {
        document.getElementById('loading-card').style.display = 'none';
    }
    
    hideResultCard() {
        document.getElementById('result-card').style.display = 'none';
    }
    
    showToast(message, type = 'info') {
        const toast = document.getElementById('toast');
        const toastBody = document.getElementById('toast-body');
        const toastTitle = document.getElementById('toast-title');
        
        toastBody.textContent = message;
        toastTitle.textContent = type.charAt(0).toUpperCase() + type.slice(1);
        
        toast.className = `toast toast-${type}`;
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
    }
    
    onTabChange(tabId) {
        if (tabId === 'dashboard-tab') {
            this.updateDashboard();
        } else if (tabId === 'model-tab') {
            this.loadModelInfo();
        }
    }
    
    startAutoRefresh() {
        // Refresh dashboard every 30 seconds
        setInterval(() => {
            const activeTab = document.querySelector('.tab-pane.active').id;
            if (activeTab === 'dashboard-tab') {
                this.updateDashboard();
            }
        }, 30000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SentimentApp();
});
