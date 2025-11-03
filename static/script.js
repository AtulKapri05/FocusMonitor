// Elements
const modeToggle = document.getElementById('mode-toggle');
const startBtn = document.getElementById('start-btn');
const focusScoreEl = document.getElementById('focus-score');
const alertStatusEl = document.getElementById('alert-status');
const toast = document.getElementById('toast');
const chartCanvas = document.getElementById('focusChart');

// Theme toggle
modeToggle.addEventListener('change', () => {
  document.body.classList.toggle('light-mode', modeToggle.checked);
});

// Chart setup (30 points sliding)
const MAX_POINTS = 30;
const ctx = chartCanvas.getContext('2d');
let dataArr = new Array(MAX_POINTS).fill(100);
const labels = new Array(MAX_POINTS).fill('');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels,
    datasets: [{
      data: dataArr,
      borderColor: '#00a8ff',
      backgroundColor: 'rgba(0,168,255,0.12)',
      tension: 0.35,
      pointRadius: 3,
      pointBackgroundColor: '#00a8ff',
      fill: true
    }]
  },
  options: {
    animation: { duration: 400 },
    scales: {
      y: { min: 0, max: 100, ticks: { color: '#9aa3ad' }, grid: { color: 'rgba(255,255,255,0.03)' } },
      x: { display: false }
    },
    plugins: { legend: { display: false } }
  }
});

// Start button (optional: used to warm up stream)
startBtn?.addEventListener('click', () => {
  // just visual; video stream already served by Flask/ <img src="/video_feed">
  startBtn.disabled = true;
  startBtn.textContent = 'Tracking…';
});

// Toast helper
function showToast(msg, ms=2200){
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(()=> toast.classList.remove('show'), ms);
}

// Fetch score from Flask and update UI + chart
async function fetchScoreAndUpdate(){
  try{
    const res = await fetch('/get_score', {cache: 'no-store'});
    if(!res.ok) throw new Error('no score');
    const json = await res.json();
    const score = Number(json.score) || 0;

    // update score text & alert status
    focusScoreEl.textContent = `${score}%`;
    if(score > 85){
      alertStatusEl.textContent = 'Fully Focused 🔥';
      alertStatusEl.className = 'big green';
    } else if(score > 60){
      alertStatusEl.textContent = 'Slightly Distracted 😐';
      alertStatusEl.className = 'big warn';
    } else {
      alertStatusEl.textContent = 'Drowsy / Unfocused 😴';
      alertStatusEl.className = 'big danger';
      showToast('⚠ Low attention detected', 1800);
    }

    // update chart (sliding window)
    dataArr.push(score);
    if(dataArr.length > MAX_POINTS) dataArr.shift();
    chart.data.datasets[0].data = dataArr;
    chart.update();
  }catch(err){
    console.error('score error', err);
  }
}

// start polling every 1s (live feel)
setInterval(fetchScoreAndUpdate, 1000);

// initial fetch
fetchScoreAndUpdate();
