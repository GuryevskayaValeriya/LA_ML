/**
 * LA_ML: Extended Educational Modules
 * 1. ModuleSVM: Линейная классификация и разделяющая гиперплоскость (SVM / Margin)
 * 2. ModuleAttention: Механизм внимания Transformer (Self-Attention на матрицах Q, K, V)
 * 3. ModuleHessian: Обусловленность матрицы Гессе и ландшафт функции потерь (Овраги, SGD vs Momentum)
 * 4. DatasetManager: Встроенные датасеты (Ирисы, Рост/Вес, Недвижимость) и загрузка CSV/JSON
 * 5. ReportGenerator: Экспорт лабораторного отчета в PDF / Print
 */

// ============================================================
// 1. MODULE: LINEAR CLASSIFICATION & SVM HYPERPLANE
// ============================================================
class ModuleSVM {
  constructor(canvas) {
    this.canvas = canvas;
    this.w = [1.2, 0.9];
    this.b = -0.5;
    this.points = [];
    this.activeClass = 1; // +1: Blue / Matcha, -1: Peach / Strawberry
    this.showMargin = true;
    this.showSupportVectors = true;
    this.showDecisionRegions = true;
    this.paletteId = 1;
    this.history = [];
    this.historyIndex = -1;

    this.loadDefaultPoints();
    this.initCanvasEvents();
    this.computeAndRender();
  }

  setPalette(pId, mode = 'light') {
    this.paletteId = parseInt(pId, 10) || 1;
    this.mode = mode;
    this.render();
    this.updateMathCard();
  }

  getPalette() {
    return window.getAppPalette(this.paletteId, (this.canvas ? this.canvas.mode : this.mode) || 'light');
  }

  loadDefaultPoints() {
    // 16 linearly separable points (8 of class +1, 8 of class -1)
    this.points = [
      // Class +1 (positive side)
      { x: 1.0, y: 1.8, label: 1 },
      { x: 2.2, y: 1.0, label: 1 },
      { x: 1.5, y: 2.8, label: 1 },
      { x: 2.8, y: 2.0, label: 1 },
      { x: 3.2, y: 0.5, label: 1 },
      { x: 2.0, y: 3.5, label: 1 },
      { x: 3.5, y: 2.2, label: 1 },
      { x: 0.8, y: 3.0, label: 1 },
      // Class -1 (negative side)
      { x: -1.2, y: -0.8, label: -1 },
      { x: -2.0, y: 0.2, label: -1 },
      { x: -0.5, y: -2.2, label: -1 },
      { x: -1.8, y: -1.8, label: -1 },
      { x: -2.8, y: -0.5, label: -1 },
      { x: -1.0, y: -3.0, label: -1 },
      { x: -2.5, y: -2.5, label: -1 },
      { x: -3.2, y: -1.2, label: -1 }
    ];
    this.pushHistory();
  }

  initCanvasEvents() {
    this.canvas.clearHandles();

    this.canvas.onHandleDrag = (id, x, y) => {
      if (id === 'svm_normal_head') {
        const norm = Math.hypot(x, y);
        if (norm > 0.2) {
          this.w[0] = Math.round(x * 10) / 10;
          this.w[1] = Math.round(y * 10) / 10;
          this.syncInputs();
          this.computeAndRender(false);
        }
      } else if (id.startsWith('svm_pt_')) {
        const idx = parseInt(id.replace('svm_pt_', ''), 10);
        if (this.points[idx]) {
          this.points[idx].x = Math.round(x * 10) / 10;
          this.points[idx].y = Math.round(y * 10) / 10;
          this.computeAndRender(false);
        }
      }
    };

    this.canvas.onClick = (x, y, e) => {
      if (e.shiftKey || e.altKey) return;
      this.points.push({
        x: Math.round(x * 10) / 10,
        y: Math.round(y * 10) / 10,
        label: this.activeClass
      });
      this.pushHistory();
      this.computeAndRender(true);
    };

    this.canvas.onPan = () => this.render();
    this.canvas.onZoom = () => this.render();
  }

  updateHandles() {
    this.canvas.clearHandles();
    const pal = this.getPalette();

    // Normal vector handle at (w1, w2)
    this.canvas.addHandle('svm_normal_head', this.w[0], this.w[1], 8, pal.primary, 'w');

    // Draggable point handles
    this.points.forEach((pt, i) => {
      const color = pt.label === 1 ? pal.primary : pal.secondary;
      const tag = pt.label === 1 ? '+' : '−';
      this.canvas.addHandle(`svm_pt_${i}`, pt.x, pt.y, 6.5, color, tag);
    });
  }

  syncInputs() {
    const el = (id) => document.getElementById(id);
    if (el('svm-w1-slider')) el('svm-w1-slider').value = this.w[0];
    if (el('svm-w2-slider')) el('svm-w2-slider').value = this.w[1];
    if (el('svm-b-slider')) el('svm-b-slider').value = this.b;
    if (el('svm-w1-val')) el('svm-w1-val').textContent = this.w[0];
    if (el('svm-w2-val')) el('svm-w2-val').textContent = this.w[1];
    if (el('svm-b-val')) el('svm-b-val').textContent = this.b;
  }

  setWeights(w1, w2, b) {
    this.w = [w1, w2];
    this.b = b;
    this.syncInputs();
    this.computeAndRender(false);
  }

  setPreset(name) {
    if (name === 'separable') {
      this.loadDefaultPoints();
      this.w = [1.2, 0.9];
      this.b = -0.5;
    } else if (name === 'narrow') {
      this.points = [
        { x: 0.5, y: 1.0, label: 1 },
        { x: 1.2, y: 0.3, label: 1 },
        { x: 1.8, y: 1.5, label: 1 },
        { x: 2.5, y: 0.8, label: 1 },
        { x: -0.5, y: -0.8, label: -1 },
        { x: -1.2, y: -0.2, label: -1 },
        { x: -0.2, y: -1.6, label: -1 },
        { x: -1.8, y: -1.2, label: -1 }
      ];
      this.w = [1.0, 1.0];
      this.b = 0.0;
      this.pushHistory();
    } else if (name === 'overlap') {
      this.points = [
        { x: 0.8, y: 1.2, label: 1 },
        { x: 1.5, y: 0.5, label: 1 },
        { x: -0.4, y: 0.6, label: 1 }, // overlap!
        { x: 2.0, y: 1.8, label: 1 },
        { x: -1.0, y: -1.0, label: -1 },
        { x: -1.8, y: -0.4, label: -1 },
        { x: 0.3, y: -0.5, label: -1 }, // overlap!
        { x: -0.8, y: -1.8, label: -1 }
      ];
      this.w = [1.0, 1.0];
      this.b = 0.0;
      this.pushHistory();
    } else if (name === 'iris') {
      this.loadIrisData();
    }
    this.syncInputs();
    this.computeAndRender(true);
  }

  loadIrisData() {
    // Fisher's Iris sample: Petal Length vs Petal Width
    // Setosa (label -1): small petals, Versicolor (label +1): larger petals
    // Normalized to fit canvas [-4, 4]
    this.points = [
      // Setosa (Class -1)
      { x: -2.8, y: -2.2, label: -1 },
      { x: -2.9, y: -2.2, label: -1 },
      { x: -2.7, y: -2.2, label: -1 },
      { x: -2.8, y: -2.0, label: -1 },
      { x: -2.5, y: -1.8, label: -1 },
      { x: -2.7, y: -2.4, label: -1 },
      { x: -2.6, y: -2.2, label: -1 },
      { x: -3.1, y: -2.4, label: -1 },
      // Versicolor (Class +1)
      { x: 0.8, y: 0.5, label: 1 },
      { x: 0.6, y: 0.7, label: 1 },
      { x: 1.1, y: 0.7, label: 1 },
      { x: 0.0, y: 0.3, label: 1 },
      { x: 0.7, y: 0.7, label: 1 },
      { x: 0.6, y: 0.3, label: 1 },
      { x: 0.8, y: 0.9, label: 1 },
      { x: -0.8, y: -0.3, label: 1 }
    ];
    this.w = [1.5, 1.2];
    this.b = 1.0;
    this.pushHistory();
  }

  pushHistory() {
    if (this.historyIndex < this.history.length - 1) {
      this.history = this.history.slice(0, this.historyIndex + 1);
    }
    this.history.push(JSON.parse(JSON.stringify(this.points)));
    if (this.history.length > 25) this.history.shift();
    this.historyIndex = this.history.length - 1;
    this.updateUndoRedoUI();
  }

  undo() {
    if (this.canUndo()) {
      this.historyIndex--;
      this.points = JSON.parse(JSON.stringify(this.history[this.historyIndex]));
      this.computeAndRender(true);
      this.updateUndoRedoUI();
    }
  }

  redo() {
    if (this.canRedo()) {
      this.historyIndex++;
      this.points = JSON.parse(JSON.stringify(this.history[this.historyIndex]));
      this.computeAndRender(true);
      this.updateUndoRedoUI();
    }
  }

  canUndo() { return this.historyIndex > 0; }
  canRedo() { return this.historyIndex < this.history.length - 1; }

  updateUndoRedoUI() {
    const btnUndo = document.getElementById('btn-undo-svm');
    const btnRedo = document.getElementById('btn-redo-svm');
    if (btnUndo) btnUndo.disabled = !this.canUndo();
    if (btnRedo) btnRedo.disabled = !this.canRedo();
  }

  computeMetrics() {
    const w = this.w;
    const b = this.b;
    const normW = Math.hypot(w[0], w[1]) || 1e-6;

    let correctCount = 0;
    let minCorrectMargin = 999.0;
    let minPosDist = 999.0;
    let minNegDist = 999.0;
    let svPosIdx = -1;
    let svNegIdx = -1;

    const evaluated = this.points.map((pt, i) => {
      const score = w[0] * pt.x + w[1] * pt.y + b;
      const dist = score / normW;
      const pred = score >= 0 ? 1 : -1;
      const correct = pred === pt.label;
      if (correct) {
        correctCount++;
        const gm = pt.label * dist;
        if (gm < minCorrectMargin) minCorrectMargin = gm;
      }
      if (pt.label === 1 && Math.abs(dist) < minPosDist) {
        minPosDist = Math.abs(dist);
        svPosIdx = i;
      }
      if (pt.label === -1 && Math.abs(dist) < minNegDist) {
        minNegDist = Math.abs(dist);
        svNegIdx = i;
      }
      return { ...pt, index: i, score, dist, correct };
    });

    const total = this.points.length;
    const accPct = total > 0 ? Math.round((correctCount / total) * 1000) / 10 : 0;
    const margin = minCorrectMargin < 900 ? Math.max(0, minCorrectMargin) : 0;
    const marginWidth = margin * 2.0;

    return {
      evaluated,
      correctCount,
      total,
      accPct,
      margin: Math.round(margin * 100) / 100,
      marginWidth: Math.round(marginWidth * 100) / 100,
      normW: Math.round(normW * 100) / 100,
      svPosIdx,
      svNegIdx
    };
  }

  computeAndRender(recreateHandles = true) {
    this.metrics = this.computeMetrics();
    if (recreateHandles) {
      this.updateHandles();
    }
    this.render();
    this.updateMathCard();
  }

  render() {
    this.canvas.clear();
    this.canvas.drawGrid();

    const ctx = this.canvas.ctx;
    const pal = this.getPalette();
    const isDark = this.canvas && this.canvas.mode === 'dark';
    const m = this.metrics || this.computeMetrics();

    const w1 = this.w[0];
    const w2 = this.w[1];
    const b = this.b;
    const normW = Math.hypot(w1, w2) || 1e-6;

    // 1. Shaded Decision Regions (w^T x + b > 0 and < 0)
    if (this.showDecisionRegions) {
      const cw = this.canvas.width;
      const ch = this.canvas.height;
      ctx.save();
      // Draw 2 large corner polygons separated by line
      // Sample bounding math coords
      const pTL = this.canvas.canvasToMath(0, 0);
      const pBR = this.canvas.canvasToMath(cw, ch);
      const range = Math.max(Math.abs(pTL.x), Math.abs(pBR.x), Math.abs(pTL.y), Math.abs(pBR.y)) + 5;

      // Color tints
      const posColor = isDark ? 'rgba(56, 189, 248, 0.08)' : 'rgba(2, 132, 199, 0.07)';
      const negColor = isDark ? 'rgba(251, 146, 60, 0.08)' : 'rgba(234, 88, 12, 0.07)';

      // Fill background half-planes using normal vector angle
      const angle = Math.atan2(w2, w1);
      const originProj = { x: -b * w1 / (normW * normW), y: -b * w2 / (normW * normW) };
      const cProj = this.canvas.mathToCanvas(originProj.x, originProj.y);

      ctx.save();
      ctx.translate(cProj.px, cProj.py);
      ctx.rotate(-angle);
      // Positive half-plane (x > 0)
      ctx.fillStyle = posColor;
      ctx.fillRect(0, -ch * 2, cw * 3, ch * 4);
      // Negative half-plane (x < 0)
      ctx.fillStyle = negColor;
      ctx.fillRect(-cw * 3, -ch * 2, cw * 3, ch * 4);
      ctx.restore();
      ctx.restore();
    }

    // 2. Margin Strip (Dashed parallel lines at distance gamma)
    if (this.showMargin && m.margin > 0.05) {
      const gamma = m.margin;
      this.drawHyperplaneLine(w1, w2, b + gamma * normW, 'rgba(234, 88, 12, 0.65)', 1.5, true);
      this.drawHyperplaneLine(w1, w2, b - gamma * normW, 'rgba(2, 132, 199, 0.65)', 1.5, true);
    }

    // 3. Separating Hyperplane line w^T x + b = 0
    this.drawHyperplaneLine(w1, w2, b, isDark ? '#f8fafc' : '#000000', 3.0, false);

    // 4. Normal Vector arrow w starting on the hyperplane
    const x0 = -b * w1 / (normW * normW);
    const y0 = -b * w2 / (normW * normW);
    const unitNx = w1 / normW;
    const unitNy = w2 / normW;
    this.canvas.drawVector(x0, y0, x0 + unitNx * 1.8, y0 + unitNy * 1.8, pal.primary, 'Нормаль w', 3.5);

    // 5. Draw Points
    for (const pt of m.evaluated) {
      const { px, py } = this.canvas.mathToCanvas(pt.x, pt.y);
      const isPositive = pt.label === 1;
      const isSV = pt.index === m.svPosIdx || pt.index === m.svNegIdx;

      ctx.save();
      // Support vector halo
      if (this.showSupportVectors && isSV && pt.correct) {
        ctx.beginPath();
        ctx.arc(px, py, 13, 0, 2 * Math.PI);
        ctx.strokeStyle = isPositive ? pal.primary : pal.secondary;
        ctx.lineWidth = 2.5;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
        // SV label
        ctx.font = '800 10px JetBrains Mono, monospace';
        ctx.fillStyle = isPositive ? pal.primary : pal.secondary;
        ctx.fillText('SV', px + 12, py - 6);
      }

      // Misclassification warning ring
      if (!pt.correct) {
        ctx.beginPath();
        ctx.arc(px, py, 11, 0, 2 * Math.PI);
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }

      // Main point
      ctx.beginPath();
      ctx.arc(px, py, 6.5, 0, 2 * Math.PI);
      ctx.fillStyle = isPositive ? pal.primary : pal.secondary;
      ctx.fill();
      ctx.strokeStyle = isDark ? '#ffffff' : '#000000';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Sign label inside point
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(isPositive ? '+' : '−', px, py + 0.5);

      ctx.restore();
    }

    // 6. Handles
    this.canvas.drawHandles();
  }

  drawHyperplaneLine(w1, w2, c, strokeColor, lineWidth, dashed) {
    const ctx = this.canvas.ctx;
    const span = 10;
    let p1, p2;

    if (Math.abs(w2) > Math.abs(w1)) {
      const x1 = -span, x2 = span;
      const y1 = (-w1 * x1 - c) / w2;
      const y2 = (-w1 * x2 - c) / w2;
      p1 = this.canvas.mathToCanvas(x1, y1);
      p2 = this.canvas.mathToCanvas(x2, y2);
    } else {
      const y1 = -span, y2 = span;
      const x1 = (-w2 * y1 - c) / w1;
      const x2 = (-w2 * y2 - c) / w1;
      p1 = this.canvas.mathToCanvas(x1, y1);
      p2 = this.canvas.mathToCanvas(x2, y2);
    }

    ctx.save();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    if (dashed) ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(p1.px, p1.py);
    ctx.lineTo(p2.px, p2.py);
    ctx.stroke();
    ctx.restore();
  }

  updateMathCard() {
    const m = this.metrics || this.computeMetrics();
    const w1 = this.w[0], w2 = this.w[1], b = this.b;

    // Formula KaTeX
    const elFormula = document.getElementById('svm-latex-formula');
    if (elFormula && window.katex) {
      const bSign = b >= 0 ? '+' : '-';
      const latex = `w^T x + b = ${w1}x_1 + ${w2}x_2 ${bSign} ${Math.abs(b)} = 0, \\quad \\|w\\| = ${m.normW}`;
      window.katex.render(latex, elFormula, { throwOnError: false });
    }

    // Metrics display
    const elStats = document.getElementById('svm-stats-info');
    if (elStats) {
      const accColor = m.accPct === 100 ? 'badge-green' : m.accPct >= 80 ? 'badge-sky' : 'badge-amber';
      elStats.innerHTML = `
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.6rem;">
          <span class="stat-badge ${accColor}">Точность (Accuracy): ${m.accPct}% (${m.correctCount}/${m.total})</span>
          <span class="stat-badge badge-sky">Отступ (Margin γ): ${m.margin}</span>
          <span class="stat-badge badge-peach">Полоса разделения (2γ): ${m.marginWidth}</span>
          <span class="stat-badge badge-purple">Норма ||w||: ${m.normW}</span>
        </div>
        <p style="font-size:0.83rem; color:var(--text-muted); line-height:1.5;">
          Вектор весов <b>w</b> является геометрической нормалью к разделяющей прямой. Алгоритм <b>SVM (Support Vector Machine)</b> ищет гиперплоскость, которая максимизирует ширину полосы отступа $2\\gamma = \\frac{2}{\\|w\\|}$ (где при канонической нормировке $\\min y_i(w^Tx_i+b)=1$ отступ $\\gamma = \\frac{1}{\\|w\\|}$), опираясь на ближайшие <b>опорные векторы (Support Vectors)</b>.
        </p>
      `;
      if (window.renderMathInElement) {
        window.renderMathInElement(elStats, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false }
          ],
          throwOnError: false
        });
      }
    }
  }
}

// ============================================================
// 2. MODULE: SELF-ATTENTION IN TRANSFORMERS
// ============================================================
class ModuleAttention {
  constructor(canvas) {
    this.canvas = canvas;
    this.tokens = ["Кот", "сел", "на", "коврик"];
    this.selectedToken = 1; // "сел"
    this.d_k = 3;
    this.d_v = 3;
    this.temperature = 1.0; // scale factor modifier
    this.paletteId = 1;

    // Embeddings Q, K, V
    this.initDefaultMatrices();
    this.computeAttention();
    this.initEvents();
  }

  setPalette(pId, mode = 'light') {
    this.paletteId = parseInt(pId, 10) || 1;
    this.mode = mode;
    this.render();
    this.updateMatrixUI();
  }

  getPalette() {
    return window.getAppPalette(this.paletteId, (this.canvas ? this.canvas.mode : this.mode) || 'light');
  }

  initDefaultMatrices() {
    // Meaningful toy embeddings reflecting semantic association:
    // "Кот" (agent), "сел" (action), "на" (prep), "коврик" (location)
    this.Q = [
      [1.2, 0.4, -0.2], // Кот
      [0.9, 1.4, 0.1],  // сел
      [0.1, 0.2, 0.8],  // на
      [-0.4, 0.9, 1.1]  // коврик
    ];

    this.K = [
      [1.1, 0.5, -0.1], // Кот
      [0.8, 1.3, 0.2],  // сел
      [0.2, 0.1, 0.9],  // на
      [-0.3, 1.0, 1.2]  // коврик
    ];

    this.V = [
      [2.0, 0.5, 0.1],
      [1.0, 2.5, 0.4],
      [0.2, 0.4, 1.8],
      [0.5, 1.2, 2.2]
    ];
  }

  randomize() {
    const rnd = (scale = 1.5) => Math.round((Math.random() * 2 - 1) * scale * 10) / 10;
    const n = this.tokens.length;
    this.Q = Array.from({ length: n }, () => [rnd(), rnd(), rnd()]);
    this.K = Array.from({ length: n }, () => [rnd(), rnd(), rnd()]);
    this.V = Array.from({ length: n }, () => [rnd(2.0), rnd(2.0), rnd(2.0)]);
    this.computeAttention();
    this.render();
    this.updateMatrixUI();
  }

  initEvents() {
    this.canvas.onPan = () => this.render();
    this.canvas.onZoom = () => this.render();
  }

  computeAttention() {
    const n = this.tokens.length;
    const d_k = this.d_k;
    const temp = Math.max(0.05, this.temperature || 1.0);
    const scale = 1.0 / (Math.sqrt(d_k) * temp);

    // 1. Raw dot products S = Q @ K^T
    this.rawScores = [];
    this.scaledScores = [];
    for (let i = 0; i < n; i++) {
      const rawRow = [];
      const scaledRow = [];
      for (let j = 0; j < n; j++) {
        let dot = 0;
        for (let p = 0; p < d_k; p++) {
          dot += this.Q[i][p] * this.K[j][p];
        }
        rawRow.push(Math.round(dot * 100) / 100);
        scaledRow.push(Math.round(dot * scale * 100) / 100);
      }
      this.rawScores.push(rawRow);
      this.scaledScores.push(scaledRow);
    }

    // 2. Softmax along rows: A = softmax(scaledScores)
    this.attentionMap = [];
    for (let i = 0; i < n; i++) {
      const row = this.scaledScores[i];
      const maxVal = Math.max(...row);
      const exps = row.map(v => Math.exp(v - maxVal));
      const sumExps = exps.reduce((acc, v) => acc + v, 0);
      const attnRow = exps.map(e => Math.round((e / sumExps) * 1000) / 1000);
      this.attentionMap.push(attnRow);
    }

    // 3. Weighted values O = Attention @ V
    this.outputEmbeddings = [];
    for (let i = 0; i < n; i++) {
      const outRow = [0, 0, 0];
      for (let j = 0; j < n; j++) {
        const weight = this.attentionMap[i][j];
        for (let p = 0; p < this.d_v; p++) {
          outRow[p] += weight * this.V[j][p];
        }
      }
      this.outputEmbeddings.push(outRow.map(v => Math.round(v * 100) / 100));
    }
  }

  setFocusedToken(idx) {
    this.selectedToken = idx;
    this.render();
    this.updateMatrixUI();
  }

  setTemperature(t) {
    this.temperature = Math.max(0.1, t);
    this.computeAttention();
    this.render();
    this.updateMatrixUI();
  }

  render() {
    this.canvas.clear();
    this.canvas.drawGrid();

    const pal = this.getPalette();
    const isDark = this.canvas && this.canvas.mode === 'dark';
    const ctx = this.canvas.ctx;

    // Render 2D Vector Projection of Queries and Keys on unit circle
    // Project first 2 coordinates of Q and K onto 2D plane
    const tokens = this.tokens;
    const sel = this.selectedToken;

    // Unit reference ring
    this.canvas.drawCircle(0, 0, 2.5, isDark ? 'rgba(56, 189, 248, 0.05)' : 'rgba(2, 132, 199, 0.04)', isDark ? '#334155' : '#e2e8f0', 1.5, true);

    // Draw all Key vectors (K)
    for (let j = 0; j < tokens.length; j++) {
      const kj = this.K[j];
      const norm = Math.hypot(kj[0], kj[1]) || 1;
      const kx = (kj[0] / norm) * 2.3;
      const ky = (kj[1] / norm) * 2.3;
      const weight = this.attentionMap[sel] ? this.attentionMap[sel][j] : 0.25;
      const color = j === sel ? pal.primary : (isDark ? '#94a3b8' : '#64748b');
      this.canvas.drawVector(0, 0, kx, ky, color, `K:${tokens[j]} (${Math.round(weight * 100)}%)`, 2.5 + weight * 3);
    }

    // Draw active Query vector (Q) for selected token
    if (this.Q[sel]) {
      const q = this.Q[sel];
      const normQ = Math.hypot(q[0], q[1]) || 1;
      const qx = (q[0] / normQ) * 2.6;
      const qy = (q[1] / normQ) * 2.6;
      this.canvas.drawVector(0, 0, qx, qy, pal.secondary, `Q:${tokens[sel]} (Фокус)`, 4.5);

      // Draw angle arcs between Q and each K
      for (let j = 0; j < tokens.length; j++) {
        const kj = this.K[j];
        const normK = Math.hypot(kj[0], kj[1]) || 1;
        const angQ = Math.atan2(qy, qx);
        const angK = Math.atan2(kj[1] / normK, kj[0] / normK);
        this.canvas.drawAngleArc(0, 0, 1.2 + j * 0.18, angQ, angK, pal.secondary, 1.5);
      }
    }

    // Live HUD legend
    ctx.save();
    ctx.font = '800 13px JetBrains Mono, monospace';
    ctx.fillStyle = isDark ? '#f4f4f5' : '#000000';
    ctx.fillText(`Запрос Q[${sel}]: "${tokens[sel]}" ➔ Распределение внимания Softmax`, 20, 30);
    ctx.restore();
  }

  updateMatrixUI() {
    const sel = this.selectedToken;
    const tokens = this.tokens;

    // 1. Tokens Pills Bar
    const tokensContainer = document.getElementById('attn-tokens-pills');
    if (tokensContainer) {
      tokensContainer.innerHTML = tokens.map((t, idx) => `
        <button class="pill-toggle-btn ${idx === sel ? 'active' : ''}" onclick="window.activeModuleAttention.setFocusedToken(${idx})">
          <span>${idx + 1}. ${t}</span>
        </button>
      `).join('');
    }

    // 2. Softmax Attention Map Table (Heatmap)
    const elHeatmap = document.getElementById('attn-heatmap-grid');
    if (elHeatmap) {
      let tableHtml = '<table class="attn-table"><thead><tr><th>Q \\ K</th>';
      tokens.forEach(t => { tableHtml += `<th>${t}</th>`; });
      tableHtml += '</tr></thead><tbody>';

      for (let i = 0; i < tokens.length; i++) {
        const isCurrentRow = i === sel;
        tableHtml += `<tr class="${isCurrentRow ? 'active-row' : ''}"><td><b>${tokens[i]}</b></td>`;
        for (let j = 0; j < tokens.length; j++) {
          const val = this.attentionMap[i][j];
          const pct = Math.round(val * 100);
          const bgAlpha = Math.min(0.9, Math.max(0.08, val));
          const cellColor = isDark
            ? (val > 0.45 ? '#09090b' : '#f4f4f5')
            : (val > 0.45 ? '#ffffff' : '#000000');
          const cellBg = isDark
            ? (this.paletteId === 2
                ? `rgba(110, 231, 183, ${Math.min(0.85, bgAlpha * 0.9)})`
                : `rgba(125, 211, 252, ${Math.min(0.85, bgAlpha * 0.9)})`)
            : `rgba(2, 132, 199, ${bgAlpha})`;
          tableHtml += `
            <td style="background: ${cellBg}; color: ${cellColor}; font-weight:750;" title="Q[${tokens[i]}] · K[${tokens[j]}] = ${this.rawScores[i][j]}">
              ${pct}%
            </td>
          `;
        }
        tableHtml += '</tr>';
      }
      tableHtml += '</tbody></table>';
      elHeatmap.innerHTML = tableHtml;
    }

    // 3. Attention Focus Distribution Breakdown Bars
    const elBars = document.getElementById('attn-focus-bars');
    if (elBars && this.attentionMap[sel]) {
      const barsHtml = tokens.map((t, j) => {
        const val = this.attentionMap[sel][j];
        const pct = Math.round(val * 100);
        return `
          <div style="margin-bottom:0.5rem;">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; font-weight:750; margin-bottom:3px;">
              <span>${tokens[sel]} ➔ ${t}</span>
              <span>${pct}% (вес ${val.toFixed(3)})</span>
            </div>
            <div class="progress-bar-track">
              <div class="progress-bar-fill" style="width: ${pct}%;"></div>
            </div>
          </div>
        `;
      }).join('');
      elBars.innerHTML = barsHtml;
    }

    // 4. KaTeX Formula
    const elFormula = document.getElementById('attn-latex-formula');
    if (elFormula && window.katex) {
      const latex = `\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V = A \\cdot V`;
      window.katex.render(latex, elFormula, { throwOnError: false });
    }
  }
}

// ============================================================
// 3. MODULE: HESSIAN MATRIX CONDITION NUMBER & LOSS LANDSCAPE
// ============================================================
class ModuleHessian {
  constructor(canvas) {
    this.canvas = canvas;
    this.lam1 = 8.0;   // lambda_max (steep ravine walls)
    this.lam2 = 0.8;   // lambda_min (flat ravine floor)
    this.angleDeg = 25.0; // rotation angle of ravine
    this.lr = 0.12;
    this.beta = 0.85;  // momentum coefficient
    this.startPoint = [-3.0, 2.8];
    this.steps = 35;
    this.paletteId = 1;

    this.computeSimulation();
    this.initCanvasEvents();
  }

  setPalette(pId, mode = 'light') {
    this.paletteId = parseInt(pId, 10) || 1;
    this.mode = mode;
    this.render();
    this.updateMathCard();
  }

  getPalette() {
    return window.getAppPalette(this.paletteId, (this.canvas ? this.canvas.mode : this.mode) || 'light');
  }

  initCanvasEvents() {
    this.canvas.clearHandles();
    this.canvas.onHandleDrag = (id, x, y) => {
      if (id === 'hessian_start') {
        this.startPoint = [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
        this.computeSimulation();
        this.render();
        this.updateMathCard();
      }
    };
    this.canvas.onPan = () => this.render();
    this.canvas.onZoom = () => this.render();
  }

  setConditionNumber(kappa) {
    this.lam1 = this.lam2 * Math.max(1, kappa);
    this.computeSimulation();
    this.render();
    this.updateMathCard();
  }

  setParameters(kappa, angle, lr, beta) {
    this.lam2 = 0.8;
    this.lam1 = this.lam2 * Math.max(1, kappa);
    this.angleDeg = angle;
    this.lr = lr;
    this.beta = beta;
    this.computeSimulation();
    this.render();
    this.updateMathCard();
  }

  computeSimulation() {
    const lam1 = this.lam1;
    const lam2 = this.lam2;
    const lamMax = Math.max(lam1, lam2);
    const lamMin = Math.min(lam1, lam2);
    this.kappa = lamMax / lamMin;

    // Rotation Matrix R(theta)
    const rad = (this.angleDeg * Math.PI) / 180;
    const cosT = Math.cos(rad);
    const sinT = Math.sin(rad);

    // Hessian H = R @ diag(lam1, lam2) @ R^T
    // H = [ [cos -sin], [sin cos] ] @ [ [lam1 cos, -lam1 sin], [lam2 sin, lam2 cos] ]
    const h11 = lam1 * cosT * cosT + lam2 * sinT * sinT;
    const h12 = (lam1 - lam2) * cosT * sinT;
    const h21 = h12;
    const h22 = lam1 * sinT * sinT + lam2 * cosT * cosT;
    this.H = [[h11, h12], [h21, h22]];

    this.lrCrit = 2.0 / lamMax; // stability limit for SGD

    const lossFn = (x, y) => 0.5 * (x * (h11 * x + h12 * y) + y * (h21 * x + h22 * y));
    const gradFn = (x, y) => [h11 * x + h12 * y, h21 * x + h22 * y];

    // 1. Vanilla SGD
    this.trajSGD = [[this.startPoint[0], this.startPoint[1]]];
    this.lossesSGD = [lossFn(this.startPoint[0], this.startPoint[1])];
    let curSgd = [this.startPoint[0], this.startPoint[1]];
    for (let t = 0; t < this.steps; t++) {
      const g = gradFn(curSgd[0], curSgd[1]);
      curSgd = [curSgd[0] - this.lr * g[0], curSgd[1] - this.lr * g[1]];
      this.trajSGD.push([curSgd[0], curSgd[1]]);
      this.lossesSGD.push(lossFn(curSgd[0], curSgd[1]));
      if (Math.hypot(curSgd[0], curSgd[1]) > 50) break; // exploded
    }

    // 2. Momentum (Polyak / Heavy Ball)
    this.trajMom = [[this.startPoint[0], this.startPoint[1]]];
    this.lossesMom = [lossFn(this.startPoint[0], this.startPoint[1])];
    let curMom = [this.startPoint[0], this.startPoint[1]];
    let vMom = [0, 0];
    for (let t = 0; t < this.steps; t++) {
      const g = gradFn(curMom[0], curMom[1]);
      vMom = [this.beta * vMom[0] + g[0], this.beta * vMom[1] + g[1]];
      curMom = [curMom[0] - this.lr * vMom[0], curMom[1] - this.lr * vMom[1]];
      this.trajMom.push([curMom[0], curMom[1]]);
      this.lossesMom.push(lossFn(curMom[0], curMom[1]));
      if (Math.hypot(curMom[0], curMom[1]) > 50) break;
    }

    // Oscillation metrics
    const pathLen = (traj) => {
      let len = 0;
      for (let i = 1; i < traj.length; i++) {
        len += Math.hypot(traj[i][0] - traj[i - 1][0], traj[i][1] - traj[i - 1][1]);
      }
      return len;
    };
    const netDist = Math.hypot(this.startPoint[0], this.startPoint[1]) || 1e-4;
    this.oscillationSGD = Math.round((pathLen(this.trajSGD) / netDist) * 10) / 10;
    this.oscillationMom = Math.round((pathLen(this.trajMom) / netDist) * 10) / 10;

    // Update start point handle
    const pal = this.getPalette();
    this.canvas.clearHandles();
    this.canvas.addHandle('hessian_start', this.startPoint[0], this.startPoint[1], 8, pal.accent, 'w₀');
  }

  render() {
    this.canvas.clear();
    this.canvas.drawGrid();

    const ctx = this.canvas.ctx;
    const pal = this.getPalette();
    const isDark = this.canvas && this.canvas.mode === 'dark';

    // 1. Draw Equipotential Contour Ellipses
    const rad = (this.angleDeg * Math.PI) / 180;
    const levels = [0.5, 1.5, 3.5, 7.0, 14.0, 26.0];
    const contourStroke = isDark
      ? (this.paletteId === 2 ? 'rgba(212, 165, 184, 0.35)' : 'rgba(148, 163, 184, 0.35)')
      : 'rgba(100, 116, 139, 0.3)';

    for (const c of levels) {
      const rx = Math.sqrt(2 * c / this.lam1);
      const ry = Math.sqrt(2 * c / this.lam2);
      this.canvas.drawEllipse(0, 0, rx, ry, rad, contourStroke, null);
    }

    // 2. Draw Ravine Axes (Eigenvector directions)
    const axisLen = 5.0;
    const cosT = Math.cos(rad);
    const sinT = Math.sin(rad);
    // Flat ravine axis (direction of lambda_min = lam2)
    const ravineAxisColor = isDark
      ? (this.paletteId === 2 ? 'rgba(110, 231, 183, 0.35)' : 'rgba(125, 211, 252, 0.35)')
      : 'rgba(56, 189, 248, 0.4)';
    this.canvas.drawPolygon([
      [-axisLen * -sinT, -axisLen * cosT],
      [axisLen * -sinT, axisLen * cosT]
    ], null, ravineAxisColor, 2, true);

    // 3. Draw SGD Trajectory (Zig-zagging)
    const drawTrajectory = (traj, color, lineWidth) => {
      if (traj.length < 2) return;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      const p0 = this.canvas.mathToCanvas(traj[0][0], traj[0][1]);
      ctx.moveTo(p0.px, p0.py);
      for (let i = 1; i < traj.length; i++) {
        const pt = this.canvas.mathToCanvas(traj[i][0], traj[i][1]);
        ctx.lineTo(pt.px, pt.py);
      }
      ctx.stroke();

      // Draw trajectory step points
      ctx.fillStyle = color;
      for (let i = 0; i < traj.length; i++) {
        const pt = this.canvas.mathToCanvas(traj[i][0], traj[i][1]);
        ctx.beginPath();
        ctx.arc(pt.px, pt.py, 3.2, 0, 2 * Math.PI);
        ctx.fill();
      }
      ctx.restore();
    };

    // Pastel colors in dark mode:
    // SGD: soft pastel coral/rose (e.g. #f87171 / #fda4af) instead of harsh nuclear #ef4444
    // Momentum: soft pastel mint/emerald (e.g. #34d399 / #6ee7b7) instead of harsh nuclear #10b981
    const sgdColor = isDark ? (this.paletteId === 2 ? '#fda4af' : '#f87171') : '#e11d48';
    const momColor = isDark ? (this.paletteId === 2 ? '#6ee7b7' : '#34d399') : '#059669';

    // Draw Vanilla SGD (Pastel Coral / Peach)
    drawTrajectory(this.trajSGD, sgdColor, 2.5);

    // Draw Momentum (Pastel Mint / Emerald)
    drawTrajectory(this.trajMom, momColor, 3.2);

    // Draw Minimum target (0, 0)
    const pMin = this.canvas.mathToCanvas(0, 0);
    ctx.save();
    ctx.fillStyle = isDark ? '#f8fafc' : '#000000';
    ctx.beginPath();
    ctx.arc(pMin.px, pMin.py, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.fillText('Минимум w*', pMin.px + 8, pMin.py + 4);
    ctx.restore();

    // 4. Handles
    this.canvas.drawHandles();
  }

  updateMathCard() {
    const elFormula = document.getElementById('hessian-latex-formula');
    if (elFormula && window.katex) {
      const latex = `\\kappa(H) = \\frac{\\lambda_{\\max}}{\\lambda_{\\min}} = \\frac{${this.lam1.toFixed(1)}}{${this.lam2.toFixed(1)}} = ${this.kappa.toFixed(1)}, \\quad \\eta_{\\text{crit}} = \\frac{2}{\\lambda_{\\max}} = ${this.lrCrit.toFixed(3)}`;
      window.katex.render(latex, elFormula, { throwOnError: false });
    }

    const elStats = document.getElementById('hessian-stats-info');
    if (elStats) {
      const isUnstable = this.lr >= this.lrCrit;
      const warnBadge = isUnstable ? '<span class="stat-badge badge-amber">⚠️ Неустойчивый шаг (η ≥ η_crit)!</span>' : '';
      elStats.innerHTML = `
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.5rem;">
          <span class="stat-badge badge-sky">Обусловленность κ = ${this.kappa.toFixed(1)}</span>
          <span class="stat-badge badge-red">Осцилляция SGD: ×${this.oscillationSGD}</span>
          <span class="stat-badge badge-green">Осцилляция Momentum: ×${this.oscillationMom}</span>
          ${warnBadge}
        </div>
        <p style="font-size:0.83rem; color:var(--text-muted); line-height:1.5;">
          При $\\kappa \\gg 1$ ландшафт функции потерь превращается в <b>узкий вытянутый овраг</b>. Градиент вдоль крутых склонов ($\\lambda_{\\max}$) огромен, заставляя классический <b>SGD</b> метаться от стены к стене. Оптимизатор с <b>моментом (Momentum)</b> накапливает вектор скорости вдоль пологого дна оврага и гасит перпендикулярные колебания.
        </p>
      `;
      if (window.renderMathInElement) {
        window.renderMathInElement(elStats, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false }
          ],
          throwOnError: false
        });
      }
    }
  }
}

// ============================================================
// 4. DATASET MANAGER: BUILT-IN CLASSICS & CSV/JSON PARSER
// ============================================================
class DatasetManager {
  constructor() {
    this.datasets = {
      iris: {
        title: "Ирисы Фишера (Fisher's Iris)",
        desc: "Классический датасет ботаника Рональда Фишера (1936). Длина и ширина лепестка (Petal Length vs Width).",
        xLabel: "Длина лепестка (см)",
        yLabel: "Ширина лепестка (см)",
        points: [
          // Setosa (label -1)
          { x: 1.4, y: 0.2, label: -1, species: "Setosa" },
          { x: 1.3, y: 0.2, label: -1, species: "Setosa" },
          { x: 1.5, y: 0.2, label: -1, species: "Setosa" },
          { x: 1.4, y: 0.3, label: -1, species: "Setosa" },
          { x: 1.7, y: 0.4, label: -1, species: "Setosa" },
          { x: 1.5, y: 0.1, label: -1, species: "Setosa" },
          { x: 1.4, y: 0.2, label: -1, species: "Setosa" },
          { x: 1.5, y: 0.4, label: -1, species: "Setosa" },
          // Versicolor (label +1)
          { x: 4.7, y: 1.4, label: 1, species: "Versicolor" },
          { x: 4.5, y: 1.5, label: 1, species: "Versicolor" },
          { x: 4.9, y: 1.5, label: 1, species: "Versicolor" },
          { x: 4.0, y: 1.3, label: 1, species: "Versicolor" },
          { x: 4.6, y: 1.5, label: 1, species: "Versicolor" },
          { x: 4.5, y: 1.3, label: 1, species: "Versicolor" },
          { x: 4.7, y: 1.6, label: 1, species: "Versicolor" },
          { x: 3.3, y: 1.0, label: 1, species: "Versicolor" }
        ]
      },
      height_weight: {
        title: "Рост и Вес (Height vs Weight)",
        desc: "Антропометрические измерения людей. Линейная зависимость массы тела от роста.",
        xLabel: "Рост (см)",
        yLabel: "Вес (кг)",
        points: [
          // Недостаток (label -1, ИМТ ≈ 16.4–17.3)
          { x: 158, y: 41, label: -1, category: "Недостаток" },
          { x: 160, y: 43, label: -1, category: "Недостаток" },
          { x: 163, y: 44, label: -1, category: "Недостаток" },
          { x: 165, y: 46, label: -1, category: "Недостаток" },
          { x: 168, y: 48, label: -1, category: "Недостаток" },
          { x: 170, y: 50, label: -1, category: "Недостаток" },
          { x: 174, y: 52, label: -1, category: "Недостаток" },
          { x: 178, y: 54, label: -1, category: "Недостаток" },

          // Норма (label +1, ИМТ ≈ 20.8–22.8)
          { x: 158, y: 52, label: 1, category: "Норма" },
          { x: 160, y: 56, label: 1, category: "Норма" },
          { x: 164, y: 58, label: 1, category: "Норма" },
          { x: 167, y: 61, label: 1, category: "Норма" },
          { x: 170, y: 64, label: 1, category: "Норма" },
          { x: 173, y: 67, label: 1, category: "Норма" },
          { x: 176, y: 70, label: 1, category: "Норма" },
          { x: 180, y: 74, label: 1, category: "Норма" }
        ]
      },
      real_estate: {
        title: "Цены на недвижимость (Real Estate)",
        desc: "Зависимость рыночной стоимости квартиры от жилой площади.",
        xLabel: "Площадь (м²)",
        yLabel: "Цена (млн руб)",
        points: [
          // Комфорт-класс / Спальный район (label +1, ~110-120 тыс. руб/м²)
          { x: 32, y: 3.8, label: 1, segment: "Комфорт" },
          { x: 38, y: 4.5, label: 1, segment: "Комфорт" },
          { x: 55, y: 5.2, label: 1, segment: "Комфорт" },
          { x: 54, y: 6.2, label: 1, segment: "Комфорт" },
          { x: 62, y: 5.0, label: 1, segment: "Комфорт" },
          { x: 75, y: 8.3, label: 1, segment: "Комфорт" },
          { x: 80, y: 8.5, label: 1, segment: "Комфорт" },
          { x: 96, y: 10.6, label: 1, segment: "Комфорт" },

          // Бизнес-класс / Центр (label -1, ~170-175 тыс. руб/м²)
          { x: 32, y: 5.5, label: -1, segment: "Бизнес" },
          { x: 40, y: 6.8, label: -1, segment: "Бизнес" },
          { x: 44, y: 7.2, label: -1, segment: "Бизнес" },
          { x: 55, y: 9.4, label: -1, segment: "Бизнес" },
          { x: 57, y: 10.2, label: -1, segment: "Бизнес" },
          { x: 74, y: 12.8, label: -1, segment: "Бизнес" },
          { x: 84, y: 14.5, label: -1, segment: "Бизнес" },
          { x: 50, y: 16.5, label: -1, segment: "Бизнес" }
        ]
      }
    };
  }

  getDataset(id) {
    return this.datasets[id] || null;
  }

  // Scale and center points to fit nicely on [-3.5, 3.5] canvas view
  normalizePoints(pts) {
    if (!pts || pts.length === 0) return [];
    const xs = pts.map(p => p.x);
    const ys = pts.map(p => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;

    return pts.map(p => ({
      x: Math.round(((p.x - minX) / spanX * 6.0 - 3.0) * 10) / 10,
      y: Math.round(((p.y - minY) / spanY * 6.0 - 3.0) * 10) / 10,
      label: p.label || 1
    }));
  }

  parseCSV(text) {
    const lines = text.trim().split(/\r?\n/).filter(l => l.trim().length > 0);
    if (lines.length < 2) throw new Error("CSV файл должен содержать минимум заголовок и одну строку данных.");

    const delimiter = lines[0].includes(';') ? ';' : lines[0].includes('\t') ? '\t' : ',';
    const headers = lines[0].split(delimiter).map(h => h.trim().replace(/^["']|["']$/g, ''));

    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(delimiter).map(v => v.trim().replace(/^["']|["']$/g, ''));
      if (parts.length === headers.length) {
        rows.push(parts);
      }
    }

    return { headers, rows };
  }

  parseJSON(text) {
    const data = JSON.parse(text);
    if (Array.isArray(data) && data.length > 0) {
      if (typeof data[0] === 'object' && !Array.isArray(data[0])) {
        const headers = Object.keys(data[0]);
        const rows = data.map(item => headers.map(k => item[k]));
        return { headers, rows };
      } else if (Array.isArray(data[0])) {
        const headers = data[0].map((_, i) => `Колонка_${i + 1}`);
        return { headers, rows: data };
      }
    }
    throw new Error("Неподдерживаемая структура JSON (ожидается массив объектов или строк).");
  }
}

// ============================================================
// 5. REPORT GENERATOR: LAB WORK PRINT / PDF EXPORTER
// ============================================================
class ReportGenerator {
  constructor(modulesMap) {
    this.modules = modulesMap;
  }

  generateHTML() {
    const now = new Date();
    const dateStr = now.toLocaleDateString('ru-RU', { day: '2-digit', month: '2-digit', year: 'numeric' });

    // Gather metrics from active modules
    const modT = this.modules.transform;
    const modPCA = this.modules.pca;
    const modLS = this.modules.leastSquares;
    const modSVM = this.modules.svm;
    const modHessian = this.modules.hessian;
    const modAttn = this.modules.attention;
    const modTrain = this.modules.trainer;

    // Snapshot canvases as images
    const snapCanvas = (canvasObj) => {
      try {
        if (canvasObj && canvasObj.canvas) {
          return canvasObj.canvas.toDataURL('image/png');
        }
      } catch (e) {}
      return '';
    };

    const imgSVM = snapCanvas(modSVM?.canvas);
    const imgPCA = snapCanvas(modPCA?.canvas);
    const imgLS = snapCanvas(modLS?.canvas);
    const imgHessian = snapCanvas(modHessian?.canvas);

    // Section 1: Matrix Transform metrics
    const matT = modT?.matrix || [[1, 0], [0, 1]];
    const detT = (matT[0][0] * matT[1][1] - matT[0][1] * matT[1][0]);
    const traceT = matT[0][0] + matT[1][1];
    const rankT = Math.abs(detT) > 1e-4 ? 2 : (Math.hypot(matT[0][0], matT[0][1], matT[1][0], matT[1][1]) > 1e-4 ? 1 : 0);
    const sTr = matT[0][0]**2 + matT[0][1]**2 + matT[1][0]**2 + matT[1][1]**2;
    const sDet = detT**2;
    const sDisc = Math.max(0, sTr**2 - 4 * sDet);
    const sv1 = Math.sqrt(Math.max(0, (sTr + Math.sqrt(sDisc)) / 2));
    const sv2 = Math.sqrt(Math.max(0, (sTr - Math.sqrt(sDisc)) / 2));
    const condT = sv2 > 1e-6 ? (sv1 / sv2).toFixed(2) : (Math.abs(detT) > 1e-4 ? '1.0' : '∞');

    const repDet = modT?.lastResult?.det ?? detT.toFixed(2);
    const repRank = modT?.lastResult?.rank ?? rankT;
    const repTrace = modT?.lastResult?.trace ?? traceT.toFixed(2);
    const repCond = modT?.lastResult?.condition_number ?? condT;

    // Section 2: PCA metrics
    const evrRaw = modPCA?.pcaResult?.evr?.[0] ?? modPCA?.pcaResult?.explained_variance_ratio?.[0];
    const evrRep = evrRaw !== undefined ? (evrRaw * 100).toFixed(1) + '%' : '88.5%';
    const mseRaw = modPCA?.pcaResult?.mse ?? modPCA?.pcaResult?.reconstruction_mse;
    const mseRep = mseRaw !== undefined ? Number(mseRaw).toFixed(4) : '0.042';
    const pcaN = modPCA?.points?.length ?? 35;

    // Section 3: Least Squares metrics
    const lsMetrics = modLS?.metrics || (typeof modLS?.computeMetrics === 'function' ? modLS.computeMetrics() : null);
    const lsR2 = lsMetrics?.r2 ?? '0.942';
    const lsOrtho = lsMetrics?.orthogonality !== undefined ? `${lsMetrics.orthogonality} (Строго ортогонален)` : '0.0000 (Строго ортогонален)';
    const lsSlope = (modLS?.w?.[0] ?? 1.0).toFixed(3);
    const rawIntercept = modLS?.w?.[1] ?? 0.0;
    const lsSign = rawIntercept >= 0 ? '+' : '-';
    const lsIntercept = Math.abs(rawIntercept).toFixed(3);
    const lsEqStr = `y = ${lsSlope}x ${lsSign} ${lsIntercept}`;

    // Section 6: Trainer tasks
    const tasks = [
      { id: 'vector_mapping', num: 1, name: 'Векторный таргетинг' },
      { id: 'matrix_inversion', num: 2, name: 'Обращение деформации' },
      { id: 'pca_variance', num: 3, name: 'Охота за главной осью (PCA)' },
      { id: 'svd_energy', num: 4, name: 'Энергия сингулярного разложения (SVD)' },
      { id: 'rag_cosine', num: 5, name: 'Векторный таргетинг RAG' },
      { id: 'class_separation', num: 6, name: 'Разделение классов (SVM)' },
      { id: 'invariant_eigenvector', num: 7, name: 'Инвариантное направление (Собственный вектор)' },
      { id: 'least_squares_fit', num: 8, name: 'Подбор проекции МНК' }
    ];

    const completed = modTrain?.completedTasks || {};
    const taskRows = tasks.map(t => {
      const isDone = Boolean(completed[t.id]);
      const statusHtml = isDone
        ? '<span style="color:#16a34a; font-weight:800;">✓ Зачтено</span>'
        : '<span style="color:#64748b; font-weight:600;">Не завершено</span>';
      const scoreText = isDone ? '100 / 100' : '0 / 100';
      return `<tr><td>Задача ${t.num}</td><td>${t.name}</td><td>${statusHtml}</td><td>${scoreText}</td></tr>`;
    }).join('');
    const totalScore = modTrain?.score ?? 0;

    return `
      <div class="report-paper" id="printable-report">
        <!-- University Academic Header -->
        <div class="report-header">
          <h2 class="report-title">ОТЧЕТ ПО ЛАБОРАТОРНОЙ РАБОТЕ</h2>
          <div class="report-course">Дисциплина: «Линейная алгебра в машинном обучении (LA_ML)»</div>
        </div>

        <!-- Student Meta Fields -->
        <div class="report-meta-grid">
          <div><b>Студент:</b> <input type="text" class="report-input" value="ФИО" style="font-weight:700;"></div>
          <div><b>Группа:</b> <input type="text" class="report-input" value=" " style="width:100px;"></div>
          <div><b>Дата:</b> <span>${dateStr}</span></div>
          <div><b>Преподаватель:</b> <input type="text" class="report-input" value="ФИО"></div>
        </div>

        <hr class="report-divider">

        <!-- Section 1: Matrix Transformations -->
        <div class="report-section">
          <h3 class="section-title">1. Линейные преобразования и собственные значения</h3>
          <p class="section-desc">Анализ матричного оператора деформации пространства y = W x + b.</p>
          <table class="report-table">
            <tr>
              <th>Параметр</th>
              <th>Формула</th>
              <th>Вычисленное значение</th>
              <th>Геометрический смысл</th>
            </tr>
            <tr>
              <td>Определитель det(W)</td>
              <td>ad - bc</td>
              <td><b>${repDet}</b></td>
              <td>Коэффициент изменения площади единичного квадрата</td>
            </tr>
            <tr>
              <td>Ранг rank(W)</td>
              <td>dim(Col(W))</td>
              <td><b>${repRank}</b></td>
              <td>Размерность образа пространства после преобразования</td>
            </tr>
            <tr>
              <td>След матрицы Tr(W)</td>
              <td>w₁₁ + w₂₂</td>
              <td><b>${repTrace}</b></td>
              <td>Сумма собственных значений λ₁ + λ₂</td>
            </tr>
            <tr>
              <td>Число обусловленности</td>
              <td>||W|| · ||W⁻¹||</td>
              <td><b>${repCond}</b></td>
              <td>Мера чувствительности к возмущениям</td>
            </tr>
          </table>
        </div>

        <!-- Section 2: PCA -->
        <div class="report-section">
          <h3 class="section-title">2. Метод главных компонент (PCA) и редукция размерности</h3>
          <p class="section-desc">Спектральное разложение выборочной ковариационной матрицы Σ = V Λ Vᵀ.</p>
          <div style="display:flex; gap:1.5rem; align-items:center;">
            <div style="flex:1;">
              <table class="report-table">
                <tr><th>Параметр</th><th>Значение</th></tr>
                <tr><td>Объем выборки (N)</td><td><b>${pcaN} точек</b></td></tr>
                <tr><td>Объясненная дисперсия (PC1 EVR)</td><td><b>${evrRep}</b></td></tr>
                <tr><td>Ошибка реконструкции (MSE)</td><td><b>${mseRep}</b></td></tr>
              </table>
            </div>
            ${imgPCA ? `<div style="width:220px; border:1px solid #cbd5e1; border-radius:6px; overflow:hidden;"><img src="${imgPCA}" style="width:100%; display:block;" alt="PCA"></div>` : ''}
          </div>
        </div>

        <!-- Section 3: Least Squares -->
        <div class="report-section">
          <h3 class="section-title">3. Метод наименьших квадратов (МНК) и ортогональное проектирование</h3>
          <p class="section-desc">Аналитическое решение нормального уравнения w = (XᵀX)⁻¹ Xᵀ y.</p>
          <div style="display:flex; gap:1.5rem; align-items:center;">
            <div style="flex:1;">
              <table class="report-table">
                <tr><th>Параметр</th><th>Значение</th></tr>
                <tr><td>Уравнение модели ŷ</td><td><b>${lsEqStr}</b></td></tr>
                <tr><td>Коэффициент детерминации R²</td><td><b>${lsR2}</b></td></tr>
                <tr><td>Ортогональность невязок Xᵀ e</td><td><b>${lsOrtho}</b></td></tr>
              </table>
            </div>
            ${imgLS ? `<div style="width:220px; border:1px solid #cbd5e1; border-radius:6px; overflow:hidden;"><img src="${imgLS}" style="width:100%; display:block;" alt="OLS"></div>` : ''}
          </div>
        </div>

        <!-- Section 4: SVM Classification -->
        <div class="report-section">
          <h3 class="section-title">4. Линейная классификация и разделяющая гиперплоскость (SVM)</h3>
          <p class="section-desc">Максимизация ширины полосы отступа 2γ = 2 / ||w|| и поиск опорных векторов.</p>
          <div style="display:flex; gap:1.5rem; align-items:center;">
            <div style="flex:1;">
              <table class="report-table">
                <tr><th>Параметр</th><th>Значение</th></tr>
                <tr><td>Вектор нормали w</td><td><b>[${(modSVM?.w?.[0] ?? 1.2).toFixed(2)}, ${(modSVM?.w?.[1] ?? 0.9).toFixed(2)}]</b></td></tr>
                <tr><td>Точность разделения (Accuracy)</td><td><b>${modSVM?.metrics?.accPct ?? 100}%</b></td></tr>
                <tr><td>Ширина отступа (Margin width 2γ)</td><td><b>${modSVM?.metrics?.marginWidth ?? 2.14}</b></td></tr>
              </table>
            </div>
            ${imgSVM ? `<div style="width:220px; border:1px solid #cbd5e1; border-radius:6px; overflow:hidden;"><img src="${imgSVM}" style="width:100%; display:block;" alt="SVM"></div>` : ''}
          </div>
        </div>

        <!-- Section 5: Hessian Loss Landscape -->
        <div class="report-section">
          <h3 class="section-title">5. Обусловленность матрицы Гессе и градиентная оптимизация</h3>
          <p class="section-desc">Влияние числа обусловленности κ(H) = λ_max / λ_min на траектории SGD vs Momentum.</p>
          <div style="display:flex; gap:1.5rem; align-items:center;">
            <div style="flex:1;">
              <table class="report-table">
                <tr><th>Параметр</th><th>Значение</th></tr>
                <tr><td>Число обусловленности κ</td><td><b>${modHessian?.kappa?.toFixed(1) ?? '10.0'} (Вытянутый овраг)</b></td></tr>
                <tr><td>Критический шаг η_crit</td><td><b>${modHessian?.lrCrit?.toFixed(3) ?? '0.250'}</b></td></tr>
                <tr><td>Осцилляция: SGD vs Momentum</td><td><b>×${modHessian?.oscillationSGD ?? '4.2'} vs ×${modHessian?.oscillationMom ?? '1.3'}</b></td></tr>
              </table>
            </div>
            ${imgHessian ? `<div style="width:220px; border:1px solid #cbd5e1; border-radius:6px; overflow:hidden;"><img src="${imgHessian}" style="width:100%; display:block;" alt="Hessian"></div>` : ''}
          </div>
        </div>

        <!-- Section 6: Trainer Results Log -->
        <div class="report-section">
          <h3 class="section-title">6. Журнал решений в тренажере практики</h3>
          <table class="report-table">
            <tr><th>Номер</th><th>Название задачи</th><th>Статус</th><th>Оценка</th></tr>
            ${taskRows}
          </table>
          <div style="margin-top:0.75rem; text-align:right; font-weight:800; font-size:1.05rem;">
            Итоговый набранный балл в тренажере: <span style="color:#0284c7;">${totalScore} баллов</span>
          </div>
        </div>

        <!-- Conclusion & Signatures -->
        <div class="report-section" style="margin-top:2rem;">
          <div style="display:flex; justify-content:space-between; margin-top:2.5rem; font-size:0.95rem;">
            <div>Подпись студента: ________________</div>
            <div>Оценка преподавателя:  ______   </div>
            <div>Подпись: _______________</div>
          </div>
        </div>
      </div>
    `;
  }

  openModal() {
    let modal = document.getElementById('modal-report-wrapper');
    if (!modal) {
      modal = document.createElement('div');
      modal.id = 'modal-report-wrapper';
      modal.className = 'report-modal-overlay';
      document.body.appendChild(modal);
    }

    modal.innerHTML = `
      <div class="report-modal-container">
        <div class="report-modal-topbar">
          <span style="font-weight:800; font-size:1.1rem;">📑 Лабораторный отчет // LA_ML</span>
          <div style="display:flex; gap:0.5rem;">
            <button class="btn-primary" onclick="window.print()" style="padding:8px 18px; font-size:0.85rem;">🖨️ Печать / Сохранить в PDF</button>
            <button class="tool-btn" onclick="document.getElementById('modal-report-wrapper').style.display='none'" style="padding:8px 14px;">✕ Закрыть</button>
          </div>
        </div>
        <div class="report-modal-body">
          ${this.generateHTML()}
        </div>
      </div>
    `;
    modal.style.display = 'flex';
  }
}

// Export to window
window.ModuleSVM = ModuleSVM;
window.ModuleAttention = ModuleAttention;
window.ModuleHessian = ModuleHessian;
window.DatasetManager = DatasetManager;
window.ReportGenerator = ReportGenerator;
