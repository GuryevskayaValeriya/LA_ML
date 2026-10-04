/**
 * LA_ML: Interactive Educational Modules Logic
 * 
 * Блок 1: Линейная алгебра
 *   - ModuleMatMul: Пошаговое анимированное умножение матриц A x B = C
 *   - ModuleVectorGeometry: Геометрический смысл векторов (комбинации, скалярное и векторное произведение)
 *   - ModuleTransform: Линейные деформации и инвариантные собственные векторы (W v = lambda v)
 * 
 * Блок 2: Машинное обучение
 *   - ModulePCA: Метод главных компонент (2D -> 1D и 3D -> 2D редукция размерности)
 *   - ModuleLeastSquares: Линейная регрессия через МНК w = (X^T X)^(-1) X^T y
 *   - ModuleSVD: Сингулярное разложение (SVD) и низкоранговая аппроксимация (LoRA)
 *   - ModuleEmbeddings: Векторные эмбеддинги, косинусное сходство (RAG) и арифметика Word2Vec
 * 
 * Практикум
 *   - ModuleTrainer: Интерактивный тренажер задач с автопроверкой
 */

// Shared Color Palette Presets:
// 1: Нео-брутализм (Небо и Персик)
// 2: Retro Studio (Матча и Клубника)
window.APP_PALETTES = {
  1: {
    id: 1,
    name: 'Небо и Персик',
    primary: '#0284c7',   // Sky blue
    secondary: '#ea580c', // Peach / coral
    accent: '#8b5cf6',    // Purple / violet
    highlightA: '#bae6fd',
    highlightB: '#fed7aa',
    highlightC: '#fef08a',
    point: '#fed7aa',
    scatter: '#38bdf8'
  },
  2: {
    id: 2,
    name: 'Матча и Клубника',
    primary: '#059669',   // Pistachio emerald
    secondary: '#e11d48', // Strawberry pink
    accent: '#7c3aed',    // Berry plum
    highlightA: '#dcfce7',
    highlightB: '#fce7f3',
    highlightC: '#f3e8ff',
    point: '#fbcfe8',
    scatter: '#34d399'
  },
  '1-dark': {
    id: 1,
    name: 'Небо и Персик (Темная)',
    primary: '#7dd3fc',   // Soft pastel sky blue
    secondary: '#fdba74', // Soft pastel peach / warm apricot
    accent: '#c4b5fd',    // Soft pastel lavender / violet
    highlightA: 'rgba(125, 211, 252, 0.22)',
    highlightB: 'rgba(253, 186, 116, 0.22)',
    highlightC: 'rgba(254, 240, 138, 0.22)',
    point: '#fdba74',
    scatter: '#7dd3fc'
  },
  '2-dark': {
    id: 2,
    name: 'Матча и Клубника (Темная)',
    primary: '#6ee7b7',   // Soft pastel matcha / mint emerald
    secondary: '#fda4af', // Soft pastel strawberry / rose
    accent: '#d8b4fe',    // Soft pastel berry plum
    highlightA: 'rgba(110, 231, 183, 0.22)',
    highlightB: 'rgba(253, 164, 175, 0.22)',
    highlightC: 'rgba(216, 180, 254, 0.22)',
    point: '#fda4af',
    scatter: '#6ee7b7'
  }
};

window.getAppPalette = function(pId = 1, mode = 'light') {
  if (mode === 'dark') {
    return window.APP_PALETTES[`${pId}-dark`] || window.APP_PALETTES[pId] || window.APP_PALETTES[1];
  }
  return window.APP_PALETTES[pId] || window.APP_PALETTES[1];
};

// ============================================================
// 1. MODULE: STEP-BY-STEP MATRIX MULTIPLICATION (A x B = C)
// ============================================================
class ModuleMatMul {
  constructor() {
    this.dimM = 2; // rows of A
    this.dimK = 2; // cols of A / rows of B
    this.dimN = 2; // cols of B

    this.A = [[2, -1], [1, 3]];
    this.B = [[1, 2], [3, 0]];
    this.fullC = [[0, 0], [0, 0]];

    this.currentStep = 0; // 0 to dimM * dimN
    this.isPlaying = false;
    this.timer = null;
    this.speed = 1200;
    this.paletteId = 1;

    this.computeFullC();
  }

  setPalette(pId) {
    this.paletteId = parseInt(pId, 10) || 1;
    this.render();
  }

  setDimensions(m, k, n) {
    this.pause();
    this.dimM = m;
    this.dimK = k;
    this.dimN = n;

    // Generate neat small integer matrices
    this.A = Array.from({ length: m }, () =>
      Array.from({ length: k }, () => Math.floor(Math.random() * 7) - 3)
    );
    this.B = Array.from({ length: k }, () =>
      Array.from({ length: n }, () => Math.floor(Math.random() * 7) - 3)
    );

    this.currentStep = 0;
    this.computeFullC();
    this.render();
  }

  computeFullC() {
    this.fullC = Array.from({ length: this.dimM }, () => Array(this.dimN).fill(0));
    for (let i = 0; i < this.dimM; i++) {
      for (let j = 0; j < this.dimN; j++) {
        let sum = 0;
        for (let p = 0; p < this.dimK; p++) {
          sum += this.A[i][p] * this.B[p][j];
        }
        this.fullC[i][j] = sum;
      }
    }
  }

  nextStep() {
    const maxSteps = this.dimM * this.dimN;
    if (this.currentStep < maxSteps) {
      this.currentStep++;
      this.render();
    } else {
      this.pause();
    }
  }

  prevStep() {
    if (this.currentStep > 0) {
      this.currentStep--;
      this.render();
    }
  }

  reset() {
    this.pause();
    this.currentStep = 0;
    this.render();
  }

  play() {
    if (this.isPlaying) return;
    this.isPlaying = true;
    const playBtn = document.getElementById('btn-matmul-play');
    if (playBtn) playBtn.innerHTML = '⏸ Пауза';

    const loop = () => {
      const maxSteps = this.dimM * this.dimN;
      if (this.currentStep >= maxSteps) {
        this.pause();
        return;
      }
      this.nextStep();
      this.timer = setTimeout(loop, this.speed);
    };
    this.timer = setTimeout(loop, 400);
  }

  pause() {
    this.isPlaying = false;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    const playBtn = document.getElementById('btn-matmul-play');
    if (playBtn) playBtn.innerHTML = '▶ Старт';
  }

  togglePlay() {
    if (this.isPlaying) this.pause();
    else {
      if (this.currentStep >= this.dimM * this.dimN) this.currentStep = 0;
      this.play();
    }
  }

  randomize() {
    this.pause();
    for (let i = 0; i < this.dimM; i++) {
      for (let p = 0; p < this.dimK; p++) {
        this.A[i][p] = Math.floor(Math.random() * 7) - 3;
      }
    }
    for (let p = 0; p < this.dimK; p++) {
      for (let j = 0; j < this.dimN; j++) {
        this.B[p][j] = Math.floor(Math.random() * 7) - 3;
      }
    }
    this.currentStep = 0;
    this.computeFullC();
    this.render();
  }

  updateMatrixCell(matrixName, r, c, val) {
    const num = parseFloat(val) || 0;
    if (matrixName === 'A') this.A[r][c] = num;
    else if (matrixName === 'B') this.B[r][c] = num;
    this.computeFullC();
    this.render();
  }

  render() {
    const containerA = document.getElementById('matmul-matrix-a');
    const containerB = document.getElementById('matmul-matrix-b');
    const containerC = document.getElementById('matmul-matrix-c');
    const detailBox = document.getElementById('matmul-step-details');
    const progressText = document.getElementById('matmul-step-progress');

    if (!containerA || !containerB || !containerC) return;

    const maxSteps = this.dimM * this.dimN;
    const isDone = this.currentStep >= maxSteps;
    const curI = isDone ? this.dimM - 1 : Math.floor(this.currentStep / this.dimN);
    const curJ = isDone ? this.dimN - 1 : this.currentStep % this.dimN;

    if (progressText) {
      progressText.textContent = `Шаг ${Math.min(this.currentStep + 1, maxSteps)} из ${maxSteps}`;
    }

    // 1. Render Matrix A
    containerA.style.gridTemplateColumns = `repeat(${this.dimK}, 1fr)`;
    containerA.innerHTML = '';
    for (let i = 0; i < this.dimM; i++) {
      for (let p = 0; p < this.dimK; p++) {
        const cell = document.createElement('div');
        cell.className = 'matmul-cell';
        if (!isDone && i === curI) cell.classList.add('active-row');
        cell.innerHTML = `<input type="number" value="${this.A[i][p]}" data-m="A" data-r="${i}" data-c="${p}">`;
        containerA.appendChild(cell);
      }
    }

    // 2. Render Matrix B
    containerB.style.gridTemplateColumns = `repeat(${this.dimN}, 1fr)`;
    containerB.innerHTML = '';
    for (let p = 0; p < this.dimK; p++) {
      for (let j = 0; j < this.dimN; j++) {
        const cell = document.createElement('div');
        cell.className = 'matmul-cell';
        if (!isDone && j === curJ) cell.classList.add('active-col');
        cell.innerHTML = `<input type="number" value="${this.B[p][j]}" data-m="B" data-r="${p}" data-c="${j}">`;
        containerB.appendChild(cell);
      }
    }

    // 3. Render Result Matrix C
    containerC.style.gridTemplateColumns = `repeat(${this.dimN}, 1fr)`;
    containerC.innerHTML = '';
    for (let i = 0; i < this.dimM; i++) {
      for (let j = 0; j < this.dimN; j++) {
        const cellIdx = i * this.dimN + j;
        const cell = document.createElement('div');
        cell.className = 'matmul-cell result-cell';

        if (cellIdx < this.currentStep || isDone) {
          cell.textContent = this.fullC[i][j];
          cell.classList.add('filled');
        } else if (cellIdx === this.currentStep) {
          cell.textContent = '?';
          cell.classList.add('active-target');
        } else {
          cell.textContent = '·';
          cell.classList.add('empty');
        }
        containerC.appendChild(cell);
      }
    }

    // Attach input listeners
    containerA.querySelectorAll('input').forEach(inp => {
      inp.addEventListener('change', (e) => {
        this.updateMatrixCell('A', parseInt(e.target.dataset.r), parseInt(e.target.dataset.c), e.target.value);
      });
    });
    containerB.querySelectorAll('input').forEach(inp => {
      inp.addEventListener('change', (e) => {
        this.updateMatrixCell('B', parseInt(e.target.dataset.r), parseInt(e.target.dataset.c), e.target.value);
      });
    });

    // 4. Render Step Breakdown Details
    if (detailBox) {
      if (isDone) {
        detailBox.innerHTML = `
          <div class="toast success" style="margin:0; box-shadow:none;">
            🎉 <b>Умножение завершено!</b> Все элементы матрицы C вычислены через скалярные произведения строк матрицы A на столбцы матрицы B.
          </div>
        `;
      } else {
        const pairs = [];
        let sumStr = '';
        let total = 0;

        for (let p = 0; p < this.dimK; p++) {
          const aVal = this.A[curI][p];
          const bVal = this.B[p][curJ];
          const prod = aVal * bVal;
          total += prod;
          pairs.push(`
            <span class="matmul-term-chip">
              <span class="chip-a">${aVal}</span> × <span class="chip-b">${bVal}</span> = <b>${prod}</b>
            </span>
          `);
        }

        detailBox.innerHTML = `
          <div style="font-size:0.95rem; font-weight:750; margin-bottom:0.5rem;">
            Вычисление элемента <span class="stat-badge badge-green">C[${curI + 1}, ${curJ + 1}]</span>:
          </div>
          <div style="display:flex; gap:0.5rem; flex-wrap:wrap; align-items:center; margin-bottom:0.75rem;">
            Скалярное произведение строки ${curI + 1} матрицы A на столбец ${curJ + 1} матрицы B:
          </div>
          <div class="matmul-chips-row">
            ${pairs.join(' <span style="font-weight:900; font-size:1.2rem;">+</span> ')}
            <span style="font-weight:900; font-size:1.2rem;">=</span>
            <span class="matmul-res-chip">${total}</span>
          </div>
        `;
      }
    }
  }
}


// ============================================================
// 2. MODULE: GEOMETRIC MEANING OF VECTORS
// ============================================================
class ModuleVectorGeometry {
  constructor(canvas) {
    this.canvas = canvas;
    this.u = [2.0, 1.0];
    this.v = [1.0, 2.5];
    this.c1 = 1.5;
    this.c2 = -0.8;
    this.mode = 'linear_combination'; // 'linear_combination' | 'dot_product' | 'cross_product'
    this.paletteId = 1;

    this.initCanvasHandles();
  }

  setPalette(pId, mode = 'light') {
    this.paletteId = parseInt(pId, 10) || 1;
    this.modeTheme = mode;
    this.updateHandles();
    this.render();
    this.updateMathCard();
  }

  getPalette() {
    return window.getAppPalette(this.paletteId, (this.canvas ? this.canvas.mode : this.modeTheme) || 'light');
  }

  initCanvasHandles() {
    this.updateHandles();
    this.canvas.onHandleDrag = (id, x, y) => {
      if (id === 'u_vec') {
        this.u = [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
        this.syncInputs();
        this.render();
      } else if (id === 'v_vec') {
        this.v = [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
        this.syncInputs();
        this.render();
      }
    };
    this.canvas.onPan = () => this.render();
    this.canvas.onZoom = () => this.render();
  }

  updateHandles() {
    this.canvas.clearHandles();
    const pal = this.getPalette();
    this.canvas.addHandle('u_vec', this.u[0], this.u[1], 8, pal.primary, 'u');
    this.canvas.addHandle('v_vec', this.v[0], this.v[1], 8, pal.secondary, 'v');
  }

  setMode(mode) {
    this.mode = mode;
    this.render();
    this.updateMathCard();
  }

  setScalars(c1, c2) {
    this.c1 = c1;
    this.c2 = c2;
    this.render();
    this.updateMathCard();
  }

  syncInputs() {
    const el = (id) => document.getElementById(id);
    if (el('vec-ux')) el('vec-ux').value = this.u[0];
    if (el('vec-uy')) el('vec-uy').value = this.u[1];
    if (el('vec-vx')) el('vec-vx').value = this.v[0];
    if (el('vec-vy')) el('vec-vy').value = this.v[1];
    this.updateMathCard();
  }

  render() {
    this.canvas.clear();
    this.canvas.drawGrid();

    const [ux, uy] = this.u;
    const [vx, vy] = this.v;
    const ctx = this.canvas.ctx;
    const pal = this.getPalette();

    if (this.mode === 'linear_combination') {
      // 1. Draw scaled vectors c1*u and c2*v
      const c1u_x = this.c1 * ux;
      const c1u_y = this.c1 * uy;
      const c2v_x = this.c2 * vx;
      const c2v_y = this.c2 * vy;
      const wx = c1u_x + c2v_x;
      const wy = c1u_y + c2v_y;

      // Parallelogram dashed lines
      this.canvas.drawPolygon([
        [0, 0],
        [c1u_x, c1u_y],
        [wx, wy],
        [c2v_x, c2v_y]
      ], pal.highlightA, pal.primary, 1.5, true);

      // Ghost vector c2*v from tip of c1*u to w
      this.canvas.drawVector(c1u_x, c1u_y, wx, wy, pal.secondary, `c₂·v`, 2);

      // Primary vectors
      this.canvas.drawVector(0, 0, ux, uy, pal.primary, 'u', 3.5);
      this.canvas.drawVector(0, 0, vx, vy, pal.secondary, 'v', 3.5);

      // Linear combination result vector w = c1*u + c2*v
      this.canvas.drawVector(0, 0, wx, wy, pal.accent, `w = ${this.c1}u + ${this.c2}v`, 4);

    } else if (this.mode === 'dot_product') {
      // 1. Draw vectors u and v
      this.canvas.drawVector(0, 0, ux, uy, pal.primary, 'u', 3.5);
      this.canvas.drawVector(0, 0, vx, vy, pal.secondary, 'v', 3.5);

      // 2. Orthogonal projection of u onto v
      const vNormSq = vx * vx + vy * vy;
      if (vNormSq > 1e-6) {
        const dot = ux * vx + uy * vy;
        const projScalar = dot / vNormSq;
        const px = projScalar * vx;
        const py = projScalar * vy;

        // Dashed perpendicular drop line
        ctx.save();
        ctx.strokeStyle = pal.secondary;
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        const pU = this.canvas.mathToCanvas(ux, uy);
        const pProj = this.canvas.mathToCanvas(px, py);
        ctx.beginPath();
        ctx.moveTo(pU.px, pU.py);
        ctx.lineTo(pProj.px, pProj.py);
        ctx.stroke();

        // Right angle marker
        ctx.fillStyle = pal.secondary;
        ctx.beginPath();
        ctx.arc(pProj.px, pProj.py, 4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.restore();

        // Projection vector on v
        this.canvas.drawVector(0, 0, px, py, pal.accent, 'proj_v(u)', 3);
      }

      // 3. Draw angle arc theta between u and v
      const angleU = Math.atan2(uy, ux);
      const angleV = Math.atan2(vy, vx);
      const pZero = this.canvas.mathToCanvas(0, 0);
      const arcRadius = 38;

      // Compute shortest directed angle difference normalized to (-PI, PI]
      let deltaAngle = angleV - angleU;
      while (deltaAngle > Math.PI) deltaAngle -= 2 * Math.PI;
      while (deltaAngle < -Math.PI) deltaAngle += 2 * Math.PI;

      // Canvas Y is flipped, so math angle theta maps to canvas angle -theta
      const startCanvasAngle = -angleU;
      const endCanvasAngle = -(angleU + deltaAngle);
      // When math angle increases (deltaAngle > 0, CCW), canvas angle decreases (canvas CCW)
      const isCanvasCCW = deltaAngle > 0;

      ctx.save();
      ctx.beginPath();
      const arcStroke = (this.canvas && this.canvas.mode === 'dark') ? '#f4f4f5' : ((this.canvas && this.canvas.theme === 'retro-studio') ? '#4a2835' : '#000000');
      ctx.arc(pZero.px, pZero.py, arcRadius, startCanvasAngle, endCanvasAngle, isCanvasCCW);
      ctx.strokeStyle = arcStroke;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Label angle theta centered on the arc
      const midMathAngle = angleU + deltaAngle / 2;
      const textX = pZero.px + (arcRadius + 14) * Math.cos(-midMathAngle);
      const textY = pZero.py + (arcRadius + 14) * Math.sin(-midMathAngle);
      ctx.font = 'bold 12px Inter, sans-serif';
      ctx.fillStyle = arcStroke;
      ctx.fillText('θ', textX, textY);
      ctx.restore();

    } else if (this.mode === 'cross_product') {
      // Parallelogram spanned by u and v
      const wx = ux + vx;
      const wy = uy + vy;
      const det = ux * vy - uy * vx;

      const fill = det >= 0 ? pal.highlightA : pal.highlightB;
      const stroke = det >= 0 ? pal.primary : pal.secondary;

      this.canvas.drawPolygon([[0, 0], [ux, uy], [wx, wy], [vx, vy]], fill, stroke, 2.5);

      // Primary vectors
      this.canvas.drawVector(0, 0, ux, uy, pal.primary, 'u', 3.5);
      this.canvas.drawVector(0, 0, vx, vy, pal.secondary, 'v', 3.5);
    }

    this.canvas.drawHandles();
  }

  updateMathCard() {
    const [ux, uy] = this.u;
    const [vx, vy] = this.v;
    const uLen = Math.hypot(ux, uy);
    const vLen = Math.hypot(vx, vy);
    const dot = ux * vx + uy * vy;
    const det = ux * vy - uy * vx; // 2D cross product / oriented area

    let cosTheta = 0;
    let thetaDeg = 0;
    if (uLen > 1e-6 && vLen > 1e-6) {
      cosTheta = Math.max(-1, Math.min(1, dot / (uLen * vLen)));
      thetaDeg = (Math.acos(cosTheta) * 180 / Math.PI).toFixed(1);
    }

    const titleEl = document.getElementById('vec-math-title');
    const contentEl = document.getElementById('vec-math-content');
    const badgeEl = document.getElementById('vec-math-badges');

    if (!contentEl) return;

    if (this.mode === 'linear_combination') {
      if (titleEl) titleEl.textContent = 'Линейная комбинация векторов и Span';
      const wx = (this.c1 * ux + this.c2 * vx).toFixed(2);
      const wy = (this.c1 * uy + this.c2 * vy).toFixed(2);
      const isIndep = Math.abs(det) > 1e-3;

      contentEl.innerHTML = `
        <div style="font-family:var(--font-mono); font-size:1rem; margin-bottom:0.4rem;">
          <b>w</b> = ${this.c1}·u + ${this.c2}·v = [${wx}, ${wy}]
        </div>
        <p style="font-size:0.85rem; color:var(--text-muted); line-height:1.5;">
          ${isIndep ? 'Векторы <b>u</b> и <b>v</b> линейно независимы. Их линейные комбинации покрывают всю плоскость ℝ² (полный ранг = 2).' : 'Векторы <b>u</b> и <b>v</b> коллинеарны (зависимы)! Их Span схлопнут в одну прямую линию.'}
        </p>
      `;

      if (badgeEl) {
        badgeEl.innerHTML = `
          <span class="stat-badge ${isIndep ? 'badge-green' : 'badge-amber'}">${isIndep ? 'Базис ℝ² (Span = вся плоскость)' : 'Коллинеарны (Span = 1D линия)'}</span>
          <span class="stat-badge badge-cyan">det([u, v]) = ${det.toFixed(2)}</span>
        `;
      }

    } else if (this.mode === 'dot_product') {
      if (titleEl) titleEl.textContent = 'Скалярное произведение и ортогональность';
      const projLen = vLen > 1e-6 ? (dot / vLen).toFixed(2) : '0';

      let angleBadgeClass = 'badge-green';
      let angleText = 'Острый угол (u·v > 0) — векторы сонаправлены';
      if (Math.abs(dot) < 0.05) {
        angleBadgeClass = 'badge-cyan';
        angleText = 'Ортогональны! (u·v = 0, угол ровно 90°)';
      } else if (dot < 0) {
        angleBadgeClass = 'badge-red';
        angleText = 'Тупой угол (u·v < 0) — векторы направлены в разные стороны';
      }

      contentEl.innerHTML = `
        <div style="font-family:var(--font-mono); font-size:1rem; margin-bottom:0.4rem;">
          <b>u · v</b> = (${ux}·${vx}) + (${uy}·${vy}) = <b>${dot.toFixed(2)}</b>
        </div>
        <div style="font-size:0.88rem; color:var(--text-muted); line-height:1.5;">
          Длины: |u| = ${uLen.toFixed(2)}, |v| = ${vLen.toFixed(2)} | Угол θ = ${thetaDeg}° | Длина проекции на v = ${projLen}
        </div>
      `;

      if (badgeEl) {
        badgeEl.innerHTML = `
          <span class="stat-badge ${angleBadgeClass}">${angleText}</span>
          <span class="stat-badge badge-cyan">cos(θ) = ${cosTheta.toFixed(3)}</span>
        `;
      }

    } else if (this.mode === 'cross_product') {
      if (titleEl) titleEl.textContent = 'Векторное произведение и ориентированная площадь';
      const area = Math.abs(det).toFixed(2);
      const orient = det >= 0 ? 'Положительная (против часовой стрелки)' : 'Отрицательная (по часовой стрелке)';

      contentEl.innerHTML = `
        <div style="font-family:var(--font-mono); font-size:1rem; margin-bottom:0.4rem;">
          <b>|u × v|</b> = |${ux}·${vy} - ${uy}·${vx}| = <b>${area}</b>
        </div>
        <p style="font-size:0.85rem; color:var(--text-muted); line-height:1.5;">
          Площадь параллелограмма, натянутого на векторы <b>u</b> и <b>v</b>, в точности равна модулю определителя матрицы [u, v].
        </p>
      `;

      if (badgeEl) {
        badgeEl.innerHTML = `
          <span class="stat-badge badge-green">Площадь S = ${area}</span>
          <span class="stat-badge ${det >= 0 ? 'badge-cyan' : 'badge-amber'}">${orient}</span>
        `;
      }
    }
  }
}


// ============================================================
// 3. MODULE 1: LINEAR TRANSFORMATIONS & EIGENVECTORS
// ============================================================
class ModuleTransform {
  constructor(canvas) {
    this.canvas = canvas;
    this.matrix = [[1.0, 0.0], [0.0, 1.0]];
    this.bias = [0.0, 0.0];
    this.animating = false;
    this.animProgress = 1.0;
    this.showNeuralLayer = false;
    this.showEigenvectors = true; // Invariant directions highlight!

    // Sample data points for neural layer classification
    this.neuralPoints = [
      { x: -1.2, y: 1.5, label: 0 },
      { x: -1.8, y: 0.8, label: 0 },
      { x: -0.8, y: 2.0, label: 0 },
      { x: -2.0, y: 1.8, label: 0 },
      { x: 1.5, y: -1.2, label: 1 },
      { x: 2.0, y: -0.5, label: 1 },
      { x: 1.2, y: -2.0, label: 1 },
      { x: 1.8, y: -1.8, label: 1 },
    ];

    this.activation = "none";
    this.paletteId = 1;
    this.initCanvasHandles();
  }

  setPalette(pId, mode = 'light') {
    this.paletteId = parseInt(pId, 10) || 1;
    this.mode = mode;
    this.updateHandles();
    this.render();
    this.updateMathCard();
  }

  getPalette() {
    return window.getAppPalette(this.paletteId, (this.canvas ? this.canvas.mode : this.mode) || 'light');
  }

  initCanvasHandles() {
    this.updateHandles();
    this.canvas.onHandleDrag = (id, x, y) => {
      if (id === 'i_hat') {
        this.matrix[0][0] = Math.round((x - this.bias[0]) * 10) / 10;
        this.matrix[1][0] = Math.round((y - this.bias[1]) * 10) / 10;
        this.syncInputs();
        this.render();
      } else if (id === 'j_hat') {
        this.matrix[0][1] = Math.round((x - this.bias[0]) * 10) / 10;
        this.matrix[1][1] = Math.round((y - this.bias[1]) * 10) / 10;
        this.syncInputs();
        this.render();
      }
    };
    this.canvas.onPan = () => this.render();
    this.canvas.onZoom = () => this.render();
  }

  updateHandles() {
    this.canvas.clearHandles();
    const pal = this.getPalette();
    // i_hat handle at column 0 of matrix + bias
    this.canvas.addHandle('i_hat', this.matrix[0][0] + this.bias[0], this.matrix[1][0] + this.bias[1], 8, pal.primary, 'î');
    // j_hat handle at column 1 of matrix + bias
    this.canvas.addHandle('j_hat', this.matrix[0][1] + this.bias[0], this.matrix[1][1] + this.bias[1], 8, pal.secondary, 'ĵ');
  }

  setMatrix(w11, w12, w21, w22) {
    this.matrix = [[w11, w12], [w21, w22]];
    this.updateHandles();
    this.syncInputs();
    this.render();
  }

  setBias(b1, b2) {
    this.bias = [b1, b2];
    this.updateHandles();
    this.syncInputs();
    this.render();
  }

  syncInputs() {
    const el = (id) => document.getElementById(id);
    if (el('m11')) el('m11').value = this.matrix[0][0];
    if (el('m12')) el('m12').value = this.matrix[0][1];
    if (el('m21')) el('m21').value = this.matrix[1][0];
    if (el('m22')) el('m22').value = this.matrix[1][1];
    if (el('b1')) el('b1').value = this.bias[0];
    if (el('b2')) el('b2').value = this.bias[1];
    this.updateMathCard();
  }

  animateTransition(targetMatrix, duration = 800) {
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    const startMatrix = JSON.parse(JSON.stringify(this.matrix));
    const startTime = performance.now();
    this.animating = true;

    const step = (time) => {
      const elapsed = time - startTime;
      const progress = Math.min(elapsed / duration, 1.0);
      const ease = progress < 0.5 ? 4 * progress * progress * progress : 1 - Math.pow(-2 * progress + 2, 3) / 2;

      this.matrix[0][0] = startMatrix[0][0] + (targetMatrix[0][0] - startMatrix[0][0]) * ease;
      this.matrix[0][1] = startMatrix[0][1] + (targetMatrix[0][1] - startMatrix[0][1]) * ease;
      this.matrix[1][0] = startMatrix[1][0] + (targetMatrix[1][0] - startMatrix[1][0]) * ease;
      this.matrix[1][1] = startMatrix[1][1] + (targetMatrix[1][1] - startMatrix[1][1]) * ease;

      this.updateHandles();
      this.render();

      if (progress < 1.0) {
        this.animFrameId = requestAnimationFrame(step);
      } else {
        this.animating = false;
        this.animFrameId = null;
        this.syncInputs();
      }
    };
    this.animFrameId = requestAnimationFrame(step);
  }

  render() {
    this.canvas.clear();
    this.canvas.drawGrid();

    // 1. Draw deformed grid lines
    this.canvas.drawTransformedGrid(this.matrix, this.bias, 7);

    // 2. Transformed Unit Square
    const [[a, b], [c, d]] = this.matrix;
    const [b1, b2] = this.bias;

    const p0 = [b1, b2];
    const p1 = [a + b1, c + b2];
    const p2 = [a + b + b1, c + d + b2];
    const p3 = [b + b1, d + b2];

    const pal = this.getPalette();
    const det = a * d - b * c;
    let fillColor = pal.highlightA;
    let strokeColor = pal.primary;

    if (Math.abs(det) < 1e-4) {
      fillColor = 'rgba(251, 146, 60, 0.28)'; // degenerate
      strokeColor = pal.secondary;
    } else if (det < 0) {
      fillColor = 'rgba(244, 63, 94, 0.22)'; // orientation flipped
      strokeColor = '#e11d48';
    }

    this.canvas.drawPolygon([p0, p1, p2, p3], fillColor, strokeColor, 2.5);

    // 3. Invariant Eigenvectors & Eigenspaces
    const trace = a + d;
    const disc = trace * trace - 4 * det;

    if (this.showEigenvectors && disc >= 0) {
      const l1 = (trace + Math.sqrt(disc)) / 2;
      const l2 = (trace - Math.sqrt(disc)) / 2;

      const getEigenvec = (l) => {
        let vx = 1, vy = 0;
        if (Math.abs(b) > 1e-5) {
          vx = b;
          vy = l - a;
        } else if (Math.abs(c) > 1e-5) {
          vx = l - d;
          vy = c;
        } else {
          if (Math.abs(l - a) < 1e-5) { vx = 1; vy = 0; }
          else { vx = 0; vy = 1; }
        }
        const norm = Math.hypot(vx, vy) || 1;
        return [vx / norm, vy / norm];
      };

      const v1 = getEigenvec(l1);
      const v2 = getEigenvec(l2);

      // Invariant lines across viewport
      const span = 8;
      this.canvas.drawPolygon([
        [b1 - v1[0] * span, b2 - v1[1] * span],
        [b1 + v1[0] * span, b2 + v1[1] * span]
      ], null, pal.accent, 2, true);

      if (Math.abs(disc) > 1e-4) {
        this.canvas.drawPolygon([
          [b1 - v2[0] * span, b2 - v2[1] * span],
          [b1 + v2[0] * span, b2 + v2[1] * span]
        ], null, pal.primary, 2, true);
      }

      // Draw original v1 and transformed W*v1 = l1 * v1
      const scale1 = 1.3;
      this.canvas.drawVector(b1, b2, b1 + v1[0] * scale1, b2 + v1[1] * scale1, pal.accent, 'v₁', 3);
      if (Math.abs(l1) > 0.05) {
        this.canvas.drawVector(b1, b2, b1 + v1[0] * scale1 * l1, b2 + v1[1] * scale1 * l1, pal.accent, `W·v₁ (λ₁=${l1.toFixed(1)})`, 3);
      }

      if (Math.abs(disc) > 1e-4) {
        const scale2 = 1.3;
        this.canvas.drawVector(b1, b2, b1 + v2[0] * scale2, b2 + v2[1] * scale2, pal.primary, 'v₂', 3);
        if (Math.abs(l2) > 0.05) {
          this.canvas.drawVector(b1, b2, b1 + v2[0] * scale2 * l2, b2 + v2[1] * scale2, pal.primary, `W·v₂ (λ₂=${l2.toFixed(1)})`, 3);
        }
      }
    }

    // 4. Transformed Basis Vectors
    this.canvas.drawVector(b1, b2, a + b1, c + b2, pal.primary, 'W·î', 3.5);
    this.canvas.drawVector(b1, b2, b + b1, d + b2, pal.secondary, 'W·ĵ', 3.5);

    // 5. If Neural Layer mode enabled, render data points and transformations
    if (this.showNeuralLayer) {
      for (const pt of this.neuralPoints) {
        const zx = a * pt.x + b * pt.y + b1;
        const zy = c * pt.x + d * pt.y + b2;

        let ax = zx;
        let ay = zy;
        if (this.activation === "relu") {
          ax = Math.max(0, zx);
          ay = Math.max(0, zy);
        } else if (this.activation === "sigmoid") {
          ax = 1.0 / (1.0 + Math.exp(-Math.max(-20, Math.min(20, zx))));
          ay = 1.0 / (1.0 + Math.exp(-Math.max(-20, Math.min(20, zy))));
        } else if (this.activation === "tanh") {
          ax = Math.tanh(zx);
          ay = Math.tanh(zy);
        }

        const color = pt.label === 0 ? '#ea580c' : '#0284c7';
        const { px, py } = this.canvas.mathToCanvas(ax, ay);

        const ctx = this.canvas.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.arc(px, py, 6.5, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#000000';
        ctx.stroke();
        ctx.restore();
      }
    }

    // 6. Draw Draggable Handles
    this.canvas.drawHandles();
  }

  updateMathCard() {
    const [[a, b], [c, d]] = this.matrix;
    const det = a * d - b * c;
    const trace = a + d;
    const disc = trace * trace - 4 * det;

    let eigStr = "";
    if (disc >= 0) {
      const l1 = (trace + Math.sqrt(disc)) / 2;
      const l2 = (trace - Math.sqrt(disc)) / 2;
      eigStr = `\\lambda_1 = ${l1.toFixed(2)}, \\; \\lambda_2 = ${l2.toFixed(2)}`;
    } else {
      const real = (trace / 2).toFixed(2);
      const im = (Math.sqrt(-disc) / 2).toFixed(2);
      eigStr = `\\lambda_{1,2} = ${real} \\pm ${im}i \\; (\\text{Комплексные / Поворот})`;
    }

    const rank = Math.abs(det) > 1e-4 ? 2 : (Math.abs(a) + Math.abs(b) + Math.abs(c) + Math.abs(d) > 1e-4 ? 1 : 0);

    const katexEl = document.getElementById('m1-latex-matrix');
    if (katexEl) {
      if (window.katex) {
        const latex = `W = \\begin{pmatrix} ${a.toFixed(2)} & ${b.toFixed(2)} \\\\ ${c.toFixed(2)} & ${d.toFixed(2)} \\end{pmatrix}, \\quad b = \\begin{pmatrix} ${this.bias[0].toFixed(2)} \\\\ ${this.bias[1].toFixed(2)} \\end{pmatrix}`;
        window.katex.render(latex, katexEl, { throwOnError: false });
      } else {
        katexEl.innerHTML = `<span style="font-family:var(--font-mono); font-size:0.92rem; color:#0284c7;">W = [[${a.toFixed(2)}, ${b.toFixed(2)}], [${c.toFixed(2)}, ${d.toFixed(2)}]]</span>`;
      }
    }

    const detEl = document.getElementById('m1-det-info');
    if (detEl) {
      let badge = "badge-green";
      let desc = "Ориентация сохранена (пространство не вырождено)";
      if (Math.abs(det) < 1e-4) {
        badge = "badge-amber";
        desc = "Вырождение (Rank = 1)! Площадь сжата в линию, матрица необратима.";
      } else if (det < 0) {
        badge = "badge-red";
        desc = "Ориентация обращена (зеркальное отражение базиса)";
      }

      detEl.innerHTML = `
        <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom: 0.35rem; flex-wrap:wrap;">
          <span class="stat-badge ${badge}">det(W) = ${det.toFixed(2)}</span>
          <span class="stat-badge badge-cyan">Rank = ${rank}</span>
          <span class="stat-badge badge-green">Trace = ${trace.toFixed(2)}</span>
        </div>
        <p style="font-size:0.8rem; color:var(--text-muted);">${desc}</p>
      `;
    }

    // Compute condition number and cache metrics for report generator
    const sTr = a * a + b * b + c * c + d * d;
    const sDet = det * det;
    const sDisc = Math.max(0, sTr * sTr - 4 * sDet);
    const sv1 = Math.sqrt(Math.max(0, (sTr + Math.sqrt(sDisc)) / 2));
    const sv2 = Math.sqrt(Math.max(0, (sTr - Math.sqrt(sDisc)) / 2));
    const condNum = sv2 > 1e-6 ? (sv1 / sv2).toFixed(2) : 'Inf';

    this.lastResult = {
      det: parseFloat(det.toFixed(2)),
      rank: rank,
      trace: parseFloat(trace.toFixed(2)),
      condition_number: condNum
    };

    const eigLatexEl = document.getElementById('m1-latex-eigen');
    if (eigLatexEl) {
      if (window.katex) {
        window.katex.render(eigStr, eigLatexEl, { throwOnError: false });
      } else {
        eigLatexEl.textContent = eigStr;
      }
    }

    const eigInfoEl = document.getElementById('m1-eigen-info');
    if (eigInfoEl) {
      if (disc >= 0) {
        eigInfoEl.innerHTML = `
          <div style="margin-top:0.45rem;">
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.35rem; flex-wrap:wrap;">
              <span class="stat-badge badge-cyan" style="font-family:var(--font-mono); font-weight:700;">W · v = λ · v</span>
              <span class="stat-badge badge-green">Действительные инвариантные оси</span>
            </div>
            <p style="font-size:0.8rem; color:var(--text-muted); line-height:1.5;">
              <b>🎯 Инвариантные оси (пунктир на графике):</b> Любой собственный вектор <b>v</b> вдоль этих осей при умножении на матрицу <b>W</b> не поворачивается, а лишь растягивается или сжимается в <b>λ</b> раз.
            </p>
          </div>
        `;
      } else {
        eigInfoEl.innerHTML = `
          <div style="margin-top:0.45rem;">
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.35rem; flex-wrap:wrap;">
              <span class="stat-badge badge-amber" style="font-family:var(--font-mono); font-weight:700;">λ₁,₂ = a ± bi</span>
              <span class="stat-badge badge-red">Нет инвариантных прямых</span>
            </div>
            <p style="font-size:0.8rem; color:#ea580c; line-height:1.5;">
              <b>🔄 Вращение пространства:</b> Комплексные корни означают отсутствие действительных инвариантных осей на плоскости — каждый ненулевой вектор при преобразовании меняет своё направление (поворачивается).
            </p>
          </div>
        `;
      }
    }
  }
}


// ============================================================
// 4. MODULE 2: PRINCIPAL COMPONENT ANALYSIS (PCA 2D & 3D)
// ============================================================
class ModulePCA {
  constructor(canvas) {
    this.canvas = canvas;
    this.mode = '2d'; // '2d' | '3d'
    this.points = [];
    this.points3D = [];
    this.pcaResult = null;
    this.pcaResult3D = null;
    this.projectionRatio = 0.0;

    // 3D Orbit camera angles
    this.rotX = 0.35;
    this.rotY = 0.65;
    this.isOrbiting = false;
    this.lastMouseX = 0;
    this.lastMouseY = 0;
    this.paletteId = 1;
    this.history = [];
    this.historyIndex = -1;

    this.generateSampleData('correlated');
    this.generateSampleData3D();
    this.initEvents();
  }

  pushHistory() {
    if (this.mode === '3d') return;
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
    const btnUndo = document.getElementById('btn-undo-pca');
    const btnRedo = document.getElementById('btn-redo-pca');
    if (btnUndo) btnUndo.disabled = !this.canUndo();
    if (btnRedo) btnRedo.disabled = !this.canRedo();
  }

  setPalette(pId, mode = 'light') {
    this.paletteId = parseInt(pId, 10) || 1;
    this.modeTheme = mode;
    this.render();
    this.updateMathCard();
  }

  getPalette() {
    return window.getAppPalette(this.paletteId, (this.canvas ? this.canvas.mode : this.modeTheme) || 'light');
  }

  computePCA() {
    return this.computeAndRender(true);
  }

  initEvents() {
    this.canvas.canvas.addEventListener('mousedown', (e) => {
      if (this.mode === '3d') {
        this.isOrbiting = true;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (this.isOrbiting && this.mode === '3d') {
        const dx = e.clientX - this.lastMouseX;
        const dy = e.clientY - this.lastMouseY;
        this.rotY += dx * 0.01;
        this.rotX += dy * 0.01;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
        this.render();
      }
    });

    window.addEventListener('mouseup', () => {
      this.isOrbiting = false;
    });

    this.canvas.onClick = (x, y, e) => {
      if (this.mode === '3d' || e.shiftKey) return;
      this.points.push([Math.round(x * 100) / 100, Math.round(y * 100) / 100]);
      this.pushHistory();
      this.computeAndRender();
    };

    this.canvas.onHandleDrag = (id, x, y) => {
      if (this.mode === '3d') return;
      const idx = parseInt(id.replace('pt_', ''), 10);
      if (!isNaN(idx) && this.points[idx]) {
        this.points[idx] = [x, y];
        this.computeAndRender(false);
      }
    };

    this.canvas.onPan = () => this.render();
    this.canvas.onZoom = () => this.render();
  }

  setMode(mode) {
    this.mode = mode;
    this.canvas.clearHandles();
    if (mode === '2d') {
      this.computeAndRender();
    } else {
      this.computeAndRender3D();
    }
  }

  generateSampleData(type = 'correlated') {
    this.points = [];
    const n = 35;
    if (type === 'correlated') {
      for (let i = 0; i < n; i++) {
        const u = (Math.random() - 0.5) * 6;
        const v = (Math.random() - 0.5) * 1.4;
        const rad = 0.6;
        const x = u * Math.cos(rad) - v * Math.sin(rad) + 0.5;
        const y = u * Math.sin(rad) + v * Math.cos(rad) - 0.2;
        this.points.push([Math.round(x * 100) / 100, Math.round(y * 100) / 100]);
      }
    } else if (type === 'circle') {
      for (let i = 0; i < n; i++) {
        const r = Math.sqrt(Math.random()) * 2.8;
        const theta = Math.random() * 2 * Math.PI;
        this.points.push([r * Math.cos(theta), r * Math.sin(theta)]);
      }
    } else if (type === 'clusters') {
      for (let i = 0; i < n / 2; i++) {
        this.points.push([-1.8 + (Math.random() - 0.5) * 1.5, 1.2 + (Math.random() - 0.5) * 1.5]);
        this.points.push([1.8 + (Math.random() - 0.5) * 1.5, -1.2 + (Math.random() - 0.5) * 1.5]);
      }
    }
    this.pushHistory();
    this.computeAndRender();
  }

  generateSampleData3D() {
    this.points3D = [];
    const n = 42;
    for (let i = 0; i < n; i++) {
      const u = (Math.random() - 0.5) * 5.5;
      const v = (Math.random() - 0.5) * 2.8;
      const w = (Math.random() - 0.5) * 0.9;
      // Stretched along rotated 3D direction
      const x = 0.7 * u - 0.4 * v + 0.3 * w;
      const y = 0.5 * u + 0.8 * v - 0.2 * w;
      const z = -0.4 * u + 0.3 * v + 0.9 * w;
      this.points3D.push([x, y, z]);
    }
    this.computeAndRender3D();
  }

  // --- 2D PCA ---
  calculateLocalPCA() {
    const pts = this.points;
    const n = pts.length;
    if (n < 2) return;

    const getX = p => (Array.isArray(p) ? p[0] : (p && p.x !== undefined ? p.x : 0));
    const getY = p => (Array.isArray(p) ? p[1] : (p && p.y !== undefined ? p.y : 0));

    let meanX = 0, meanY = 0;
    for (const p of pts) {
      meanX += getX(p);
      meanY += getY(p);
    }
    meanX /= n;
    meanY /= n;

    let cxx = 0, cxy = 0, cyy = 0;
    for (const p of pts) {
      const dx = getX(p) - meanX;
      const dy = getY(p) - meanY;
      cxx += dx * dx;
      cxy += dx * dy;
      cyy += dy * dy;
    }
    const denom = Math.max(n - 1, 1);
    cxx /= denom;
    cxy /= denom;
    cyy /= denom;

    const tr = cxx + cyy;
    const det = cxx * cyy - cxy * cxy;
    const disc = Math.sqrt(Math.max(0, (cxx - cyy) ** 2 + 4 * cxy * cxy));

    const lambda1 = Math.max(0, (tr + disc) / 2);
    const lambda2 = Math.max(0, (tr - disc) / 2);

    let v1 = [1, 0];
    if (Math.abs(cxy) > 1e-6) {
      v1 = [lambda1 - cyy, cxy];
    } else if (cxx < cyy) {
      v1 = [0, 1];
    }
    const norm1 = Math.hypot(v1[0], v1[1]) || 1;
    v1 = [v1[0] / norm1, v1[1] / norm1];
    const v2 = [-v1[1], v1[0]];

    const totalVar = lambda1 + lambda2;
    const evr1 = totalVar > 1e-8 ? lambda1 / totalVar : 0.5;
    const evr2 = totalVar > 1e-8 ? lambda2 / totalVar : 0.5;

    const projected = [];
    let mse = 0;
    for (const p of pts) {
      const px = getX(p);
      const py = getY(p);
      const dx = px - meanX;
      const dy = py - meanY;
      const t = dx * v1[0] + dy * v1[1];
      const projX = meanX + t * v1[0];
      const projY = meanY + t * v1[1];
      projected.push([projX, projY]);
      mse += (px - projX) ** 2 + (py - projY) ** 2;
    }
    mse /= n;

    this.pcaResult = {
      mean: [meanX, meanY],
      cov: [[cxx, cxy], [cxy, cyy]],
      eigenvalues: [lambda1, lambda2],
      eigenvectors: [v1, v2],
      evr: [evr1, evr2],
      explained_variance_ratio: [evr1, evr2],
      totalVar,
      mse,
      reconstruction_mse: mse,
      projected
    };
  }

  // --- 3D PCA via Jacobi Algorithm ---
  calculate3DPCA() {
    const pts = this.points3D;
    const n = pts.length;
    let mx = 0, my = 0, mz = 0;
    for (const p of pts) {
      mx += p[0]; my += p[1]; mz += p[2];
    }
    mx /= n; my /= n; mz /= n;

    // 3x3 Covariance
    const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const p of pts) {
      const d = [p[0] - mx, p[1] - my, p[2] - mz];
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          cov[r][c] += d[r] * d[c];
        }
      }
    }
    const denom = Math.max(n - 1, 1);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) cov[r][c] /= denom;
    }

    // Jacobi 3x3
    let A = JSON.parse(JSON.stringify(cov));
    let V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

    for (let iter = 0; iter < 18; iter++) {
      let maxOff = 0, p = 0, q = 1;
      for (let i = 0; i < 3; i++) {
        for (let j = i + 1; j < 3; j++) {
          if (Math.abs(A[i][j]) > maxOff) {
            maxOff = Math.abs(A[i][j]);
            p = i; q = j;
          }
        }
      }
      if (maxOff < 1e-7) break;

      const diff = A[q][q] - A[p][p];
      let t;
      if (Math.abs(diff) < 1e-7) {
        t = 1.0;
      } else {
        const phi = diff / (2.0 * A[p][q]);
        t = 1.0 / (Math.abs(phi) + Math.sqrt(phi * phi + 1.0));
        if (phi < 0) t = -t;
      }
      const c = 1.0 / Math.sqrt(t * t + 1.0);
      const s = t * c;
      const tau = s / (1.0 + c);

      const h = t * A[p][q];
      A[p][p] -= h;
      A[q][q] += h;
      A[p][q] = 0;

      for (let j = 0; j < 3; j++) {
        if (j !== p && j !== q) {
          const g = A[j][p];
          const h2 = A[j][q];
          A[j][p] = g - s * (h2 + g * tau);
          A[j][q] = h2 + s * (g - h2 * tau);
          A[p][j] = A[j][p];
          A[q][j] = A[j][q];
        }
      }
      for (let j = 0; j < 3; j++) {
        const g = V[j][p];
        const h2 = V[j][q];
        V[j][p] = g - s * (h2 + g * tau);
        V[j][q] = h2 + s * (g - h2 * tau);
      }
    }

    const eigvals = [A[0][0], A[1][1], A[2][2]];
    const order = [0, 1, 2].sort((a, b) => eigvals[b] - eigvals[a]);
    const sortedVals = order.map(i => Math.max(0, eigvals[i]));
    const sortedVecs = order.map(i => [V[0][i], V[1][i], V[2][i]]);

    const totalVar = sortedVals[0] + sortedVals[1] + sortedVals[2] || 1;
    const evr = sortedVals.map(v => v / totalVar);

    // Project points onto principal plane (v1, v2)
    const v1 = sortedVecs[0];
    const v2 = sortedVecs[1];
    const projected3D = [];

    for (const p of pts) {
      const d = [p[0] - mx, p[1] - my, p[2] - mz];
      const dot1 = d[0] * v1[0] + d[1] * v1[1] + d[2] * v1[2];
      const dot2 = d[0] * v2[0] + d[1] * v2[1] + d[2] * v2[2];
      const px = mx + dot1 * v1[0] + dot2 * v2[0];
      const py = my + dot1 * v1[1] + dot2 * v2[1];
      const pz = mz + dot1 * v1[2] + dot2 * v2[2];
      projected3D.push([px, py, pz]);
    }

    this.pcaResult3D = {
      mean: [mx, my, mz],
      cov,
      eigenvalues: sortedVals,
      eigenvectors: sortedVecs,
      evr,
      projected3D
    };
  }

  setProjectionRatio(val) {
    this.projectionRatio = parseFloat(val);
    this.render();
    this.updateMathCard();
  }

  async computeAndRender(syncHandles = true) {
    if (this.points.length < 2) return;
    this.calculateLocalPCA();

    if (syncHandles) {
      this.canvas.clearHandles();
      for (let i = 0; i < Math.min(this.points.length, 12); i++) {
        this.canvas.addHandle(`pt_${i}`, this.points[i][0], this.points[i][1], 6, '#7dd3fc', '');
      }
    }

    this.render();
    this.updateMathCard();
  }

  computeAndRender3D() {
    this.calculate3DPCA();
    this.render();
    this.updateMathCard();
  }

  project3DToScreen(x, y, z) {
    const cosY = Math.cos(this.rotY), sinY = Math.sin(this.rotY);
    const cosX = Math.cos(this.rotX), sinX = Math.sin(this.rotX);

    // Rotate around Y
    const x1 = x * cosY + z * sinY;
    const z1 = -x * sinY + z * cosY;

    // Rotate around X
    const y2 = y * cosX - z1 * sinX;
    const z2 = y * sinX + z1 * cosX;

    // Perspective / Isometric scaling
    const scale = this.canvas.scale * 0.85;
    return {
      px: this.canvas.originX + x1 * scale,
      py: this.canvas.originY - y2 * scale,
      depth: z2
    };
  }

  render() {
    this.canvas.clear();

    if (this.mode === '2d') {
      this.canvas.drawGrid();
      if (!this.pcaResult) return;
      const { mean, eigenvectors, eigenvalues, projected } = this.pcaResult;
      const [v1, v2] = eigenvectors;
      const [l1, l2] = eigenvalues;
      const ctx = this.canvas.ctx;

      const pal = this.getPalette();
      const isDark = this.canvas && this.canvas.mode === 'dark';
      // 1. Dispersion ellipses
      const angleRad = Math.atan2(v1[1], v1[0]);
      this.canvas.drawEllipse(mean[0], mean[1], Math.sqrt(l1), Math.sqrt(l2), angleRad, pal.primary, pal.highlightA);
      this.canvas.drawEllipse(mean[0], mean[1], Math.sqrt(l1) * 2, Math.sqrt(l2) * 2, angleRad, pal.primary, pal.highlightA);

      // 2. Principal Axes Lines
      const span = 8;
      this.canvas.drawPolygon([
        [mean[0] - v1[0] * span, mean[1] - v1[1] * span],
        [mean[0] + v1[0] * span, mean[1] + v1[1] * span]
      ], null, pal.primary, 1.5, true);

      // 3. Points with projection interpolation
      for (let i = 0; i < this.points.length; i++) {
        const orig = this.points[i];
        const proj = projected[i];

        const curX = orig[0] + (proj[0] - orig[0]) * this.projectionRatio;
        const curY = orig[1] + (proj[1] - orig[1]) * this.projectionRatio;

        if (this.projectionRatio > 0.01) {
          const pOrig = this.canvas.mathToCanvas(orig[0], orig[1]);
          const pCur = this.canvas.mathToCanvas(curX, curY);
          ctx.save();
          ctx.strokeStyle = pal.secondary;
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(pOrig.px, pOrig.py);
          ctx.lineTo(pCur.px, pCur.py);
          ctx.stroke();
          ctx.restore();
        }

        const { px, py } = this.canvas.mathToCanvas(curX, curY);
        ctx.save();
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, 2 * Math.PI);
        ctx.fillStyle = pal.scatter;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = isDark ? '#f4f4f5' : '#000000';
        ctx.stroke();
        ctx.restore();
      }

      // 4. Eigenvectors
      const len1 = Math.max(Math.sqrt(l1) * 1.5, 0.5);
      const len2 = Math.max(Math.sqrt(l2) * 1.5, 0.3);
      this.canvas.drawVector(mean[0], mean[1], mean[0] + v1[0] * len1, mean[1] + v1[1] * len1, pal.primary, 'PC1 (v₁)', 3.5);
      this.canvas.drawVector(mean[0], mean[1], mean[0] + v2[0] * len2, mean[1] + v2[1] * len2, pal.secondary, 'PC2 (v₂)', 2.5);

      // 5. Mean Marker
      const pMean = this.canvas.mathToCanvas(mean[0], mean[1]);
      ctx.save();
      ctx.fillStyle = isDark ? '#f4f4f5' : '#000000';
      ctx.beginPath();
      ctx.arc(pMean.px, pMean.py, 4.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.font = 'bold 11px Inter';
      ctx.fillText(' μ', pMean.px + 5, pMean.py - 5);
      ctx.restore();

      this.canvas.drawHandles();

    } else {
      // 3D VIEWPORT RENDERING
      const pal = this.getPalette();
      const isDark = this.canvas && this.canvas.mode === 'dark';
      const ctx = this.canvas.ctx;
      const w = this.canvas.displayWidth;
      const h = this.canvas.displayHeight;

      let bg3D = '#ffffff';
      if (this.canvas.theme === 'retro-studio') {
        bg3D = isDark ? '#24151f' : '#fdfbf7';
      } else {
        bg3D = isDark ? '#18181b' : '#ffffff';
      }
      ctx.fillStyle = bg3D;
      ctx.fillRect(0, 0, w, h);

      if (!this.pcaResult3D) return;
      const { mean, eigenvectors, eigenvalues, projected3D } = this.pcaResult3D;
      const [v1, v2, v3] = eigenvectors;

      // 1. Draw 3D Axes (X, Y, Z)
      const origin3D = this.project3DToScreen(0, 0, 0);
      const pX = this.project3DToScreen(3.5, 0, 0);
      const pY = this.project3DToScreen(0, 3.5, 0);
      const pZ = this.project3DToScreen(0, 0, 3.5);

      const draw3DLine = (pA, pB, color, label) => {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(pA.px, pA.py);
        ctx.lineTo(pB.px, pB.py);
        ctx.stroke();
        if (label) {
          ctx.fillStyle = color;
          ctx.font = 'bold 12px Inter';
          ctx.fillText(label, pB.px + 4, pB.py - 4);
        }
        ctx.restore();
      };

      const axisColor = isDark ? '#94a3b8' : '#64748b';
      draw3DLine(origin3D, pX, axisColor, 'X');
      draw3DLine(origin3D, pY, axisColor, 'Y');
      draw3DLine(origin3D, pZ, axisColor, 'Z');

      // 2. Draw 2D Principal Hyperplane quad (spanned by v1 and v2)
      const plSpan = 3.5;
      const c0 = this.project3DToScreen(mean[0] - v1[0]*plSpan - v2[0]*plSpan, mean[1] - v1[1]*plSpan - v2[1]*plSpan, mean[2] - v1[2]*plSpan - v2[2]*plSpan);
      const c1 = this.project3DToScreen(mean[0] + v1[0]*plSpan - v2[0]*plSpan, mean[1] + v1[1]*plSpan - v2[1]*plSpan, mean[2] + v1[2]*plSpan - v2[2]*plSpan);
      const c2 = this.project3DToScreen(mean[0] + v1[0]*plSpan + v2[0]*plSpan, mean[1] + v1[1]*plSpan + v2[1]*plSpan, mean[2] + v1[2]*plSpan + v2[2]*plSpan);
      const c3 = this.project3DToScreen(mean[0] - v1[0]*plSpan + v2[0]*plSpan, mean[1] - v1[1]*plSpan + v2[1]*plSpan, mean[2] - v1[2]*plSpan + v2[2]*plSpan);

      ctx.save();
      ctx.fillStyle = pal.highlightA;
      ctx.strokeStyle = pal.primary;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(c0.px, c0.py);
      ctx.lineTo(c1.px, c1.py);
      ctx.lineTo(c2.px, c2.py);
      ctx.lineTo(c3.px, c3.py);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();

      // 3. Draw 3D Points with projection drop lines
      const ptsToDraw = [];
      for (let i = 0; i < this.points3D.length; i++) {
        const orig = this.points3D[i];
        const proj = projected3D[i];

        const curX = orig[0] + (proj[0] - orig[0]) * this.projectionRatio;
        const curY = orig[1] + (proj[1] - orig[1]) * this.projectionRatio;
        const curZ = orig[2] + (proj[2] - orig[2]) * this.projectionRatio;

        const screenPos = this.project3DToScreen(curX, curY, curZ);
        const origScreen = this.project3DToScreen(orig[0], orig[1], orig[2]);

        ptsToDraw.push({ screenPos, origScreen, depth: screenPos.depth });
      }

      // Sort by depth for correct 3D overlap
      ptsToDraw.sort((a, b) => b.depth - a.depth);

      for (const pt of ptsToDraw) {
        if (this.projectionRatio > 0.01) {
          ctx.save();
          ctx.strokeStyle = pal.secondary;
          ctx.lineWidth = 1;
          ctx.setLineDash([2, 2]);
          ctx.beginPath();
          ctx.moveTo(pt.origScreen.px, pt.origScreen.py);
          ctx.lineTo(pt.screenPos.px, pt.screenPos.py);
          ctx.stroke();
          ctx.restore();
        }

        ctx.save();
        ctx.beginPath();
        ctx.arc(pt.screenPos.px, pt.screenPos.py, 4.5, 0, 2 * Math.PI);
        ctx.fillStyle = pal.scatter;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = '#000000';
        ctx.stroke();
        ctx.restore();
      }

      // 4. Principal component vectors (PC1, PC2)
      const pMean3D = this.project3DToScreen(mean[0], mean[1], mean[2]);
      const pPC1 = this.project3DToScreen(mean[0] + v1[0] * 2.5, mean[1] + v1[1] * 2.5, mean[2] + v1[2] * 2.5);
      const pPC2 = this.project3DToScreen(mean[0] + v2[0] * 2.0, mean[1] + v2[1] * 2.0, mean[2] + v2[2] * 2.0);

      draw3DLine(pMean3D, pPC1, pal.primary, 'PC1');
      draw3DLine(pMean3D, pPC2, pal.secondary, 'PC2');

      // Hint on canvas
      ctx.save();
      ctx.font = '600 12px Inter';
      ctx.fillStyle = '#64748b';
      ctx.fillText('🖱️ Зажмите мышь и тяните для вращения 3D сцены', 15, h - 15);
      ctx.restore();
    }
  }

  updateMathCard() {
    const katexEl = document.getElementById('m2-latex-cov');
    const evrEl = document.getElementById('m2-evr-info');
    const statsEl = document.getElementById('m2-stats-info');

    if (this.mode === '2d') {
      if (!this.pcaResult) return;
      const { cov, eigenvalues, evr, mse } = this.pcaResult;

      if (katexEl) {
        if (window.katex) {
          const latex = `\\Sigma = \\begin{pmatrix} ${cov[0][0].toFixed(2)} & ${cov[0][1].toFixed(2)} \\\\ ${cov[1][0].toFixed(2)} & ${cov[1][1].toFixed(2)} \\end{pmatrix}`;
          window.katex.render(latex, katexEl, { throwOnError: false });
        } else {
          katexEl.innerHTML = `<span style="font-family:var(--font-mono); color:#0284c7;">Σ = [[${cov[0][0].toFixed(2)}, ${cov[0][1].toFixed(2)}], [${cov[1][0].toFixed(2)}, ${cov[1][1].toFixed(2)}]]</span>`;
        }
      }

      if (evrEl) {
        if (window.katex) {
          const latex = `EVR_1 = ${(evr[0] * 100).toFixed(1)}\\%, \\quad EVR_2 = ${(evr[1] * 100).toFixed(1)}\\%`;
          window.katex.render(latex, evrEl, { throwOnError: false });
        } else {
          evrEl.innerHTML = `<span style="font-family:var(--font-mono); color:#475569;">EVR₁ = ${(evr[0] * 100).toFixed(1)}%, EVR₂ = ${(evr[1] * 100).toFixed(1)}%</span>`;
        }
      }

      if (statsEl) {
        statsEl.innerHTML = `
          <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
            <span class="stat-badge badge-cyan">λ₁ = ${eigenvalues[0].toFixed(2)}</span>
            <span class="stat-badge badge-amber">λ₂ = ${eigenvalues[1].toFixed(2)}</span>
            <span class="stat-badge badge-green">MSE сжатия = ${(mse * this.projectionRatio).toFixed(3)}</span>
          </div>
        `;
      }
    } else {
      // 3D PCA Math
      if (!this.pcaResult3D) return;
      const { cov, eigenvalues, evr } = this.pcaResult3D;

      if (katexEl) {
        if (window.katex) {
          const latex = `\\Sigma_{3\\times 3} = \\begin{pmatrix} 
            ${cov[0][0].toFixed(1)} & ${cov[0][1].toFixed(1)} & ${cov[0][2].toFixed(1)} \\\\ 
            ${cov[1][0].toFixed(1)} & ${cov[1][1].toFixed(1)} & ${cov[1][2].toFixed(1)} \\\\ 
            ${cov[2][0].toFixed(1)} & ${cov[2][1].toFixed(1)} & ${cov[2][2].toFixed(1)} 
          \\end{pmatrix}`;
          window.katex.render(latex, katexEl, { throwOnError: false });
        } else {
          katexEl.innerHTML = `<span style="font-family:var(--font-mono); font-size:0.85rem;">Σ = 3x3 ковариационная матрица</span>`;
        }
      }

      if (evrEl) {
        const retained = ((evr[0] + evr[1]) * 100).toFixed(1);
        if (window.katex) {
          const latex = `EVR_1 = ${(evr[0]*100).toFixed(1)}\\%, \\; EVR_2 = ${(evr[1]*100).toFixed(1)}\\%, \\; EVR_3 = ${(evr[2]*100).toFixed(1)}\\%`;
          window.katex.render(latex, evrEl, { throwOnError: false });
        } else {
          evrEl.innerHTML = `EVR: PC1=${(evr[0]*100).toFixed(1)}%, PC2=${(evr[1]*100).toFixed(1)}%, PC3=${(evr[2]*100).toFixed(1)}%`;
        }
      }

      if (statsEl) {
        const retained = ((evr[0] + evr[1]) * 100).toFixed(1);
        const lost = (evr[2] * 100).toFixed(1);
        statsEl.innerHTML = `
          <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.4rem;">
            <span class="stat-badge badge-green">Сохранено дисперсии: ${retained}%</span>
            <span class="stat-badge badge-amber">Потеря информации (λ₃): ${lost}%</span>
          </div>
          <p style="font-size:0.82rem; color:var(--text-muted); line-height:1.5;">
            Проекция 3D точек на плоскость первых двух главных компонент (PC1, PC2) сжимает размерность с сохранением ${retained}% ключевой дисперсии данных.
          </p>
        `;
      }
    }
  }
}


// ============================================================
// 5. MODULE 3: LEAST SQUARES & SUBSPACE PROJECTION
// ============================================================
class ModuleLeastSquares {
  constructor(canvas) {
    this.canvas = canvas;
    this.points = [
      [-2.5, -1.8],
      [-1.5, -0.6],
      [-0.5, 0.4],
      [0.8, 1.2],
      [2.0, 2.1],
      [3.0, 3.2]
    ];
    this.fitIntercept = true;
    this.w = [1.0, 0.0];
    this.autoGDTimer = null;
    this.iterationCount = 0;
    this.paletteId = 1;
    this.statusMessage = "🎯 <b>Аналитическое решение МНК:</b> глобальный минимум найден мгновенно за 1 шаг! Градиент ∇L = 0.";
    this.statusType = "optimum";
    this.history = [];
    this.historyIndex = -1;
    this.pushHistory();
    this.initCanvasHandles();
    this.solveOLS();
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
      this.initCanvasHandles();
      this.solveOLS();
      this.updateUndoRedoUI();
    }
  }

  redo() {
    if (this.canRedo()) {
      this.historyIndex++;
      this.points = JSON.parse(JSON.stringify(this.history[this.historyIndex]));
      this.initCanvasHandles();
      this.solveOLS();
      this.updateUndoRedoUI();
    }
  }

  canUndo() { return this.historyIndex > 0; }
  canRedo() { return this.historyIndex < this.history.length - 1; }

  updateUndoRedoUI() {
    const btnUndo = document.getElementById('btn-undo-ls');
    const btnRedo = document.getElementById('btn-redo-ls');
    if (btnUndo) btnUndo.disabled = !this.canUndo();
    if (btnRedo) btnRedo.disabled = !this.canRedo();
  }

  setPalette(pId, mode = 'light') {
    this.paletteId = parseInt(pId, 10) || 1;
    this.mode = mode;
    this.initCanvasHandles();
    this.render();
    this.updateMathCard();
  }

  getPalette() {
    return window.getAppPalette(this.paletteId, (this.canvas ? this.canvas.mode : this.mode) || 'light');
  }

  initCanvasHandles() {
    this.canvas.clearHandles();
    const pal = this.getPalette();
    this.points.forEach((pt, i) => {
      this.canvas.addHandle(`ls_pt_${i}`, pt[0], pt[1], 7.5, pal.point, `${i + 1}`);
    });

    this.canvas.onHandleDrag = (id, x, y) => {
      const idx = parseInt(id.replace('ls_pt_', ''), 10);
      if (!isNaN(idx) && this.points[idx]) {
        this.points[idx] = [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
        this.solveOLS();
      }
    };

    this.canvas.onPan = () => this.render();
    this.canvas.onZoom = () => this.render();

    this.canvas.onClick = (x, y) => {
      if (this.points.length >= 16) return;
      this.points.push([Math.round(x * 10) / 10, Math.round(y * 10) / 10]);
      this.pushHistory();
      this.initCanvasHandles();
      this.solveOLS();
    };
  }

  solveOLS() {
    this.stopAutoGD();
    const pts = this.points;
    const n = pts.length;
    if (n < 2) return;

    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (const p of pts) {
      sumX += p[0];
      sumY += p[1];
      sumXY += p[0] * p[1];
      sumX2 += p[0] * p[0];
    }

    let isCollinear = false;
    if (this.fitIntercept) {
      const denom = (n * sumX2 - sumX * sumX);
      if (Math.abs(denom) > 1e-7) {
        const slope = (n * sumXY - sumX * sumY) / denom;
        const intercept = (sumY - slope * sumX) / n;
        this.w = [slope, intercept];
      } else {
        isCollinear = true;
        this.w = [0.0, sumY / n];
      }
    } else {
      if (Math.abs(sumX2) > 1e-7) {
        this.w = [sumXY / sumX2, 0.0];
      } else {
        isCollinear = true;
        this.w = [0.0, 0.0];
      }
    }

    this.iterationCount = 0;
    if (isCollinear) {
      this.statusType = "scrambled";
      this.statusMessage = "⚠️ <b>Вырожденный случай:</b> все точки лежат на одной вертикали (x = const). Прямая функции y(x) не определена (деление на ноль в МНК).";
    } else {
      this.statusType = "optimum";
      this.statusMessage = "🎯 <b>Аналитическое решение МНК:</b> глобальный минимум найден мгновенно за 1 матричный шаг через <i>(XᵀX)⁻¹Xᵀy</i>! Вектор невязок <b>e</b> строго ортогонален признакам, поэтому градиент ∇L = 0.";
    }

    this.metrics = this.computeMetrics();
    this.render();
    this.updateMathCard();
    this.updateStatusUI();
  }

  computeMetrics() {
    const pts = this.points;
    const n = pts.length;
    if (n < 2) return { r2: '1.000', orthogonality: '0.0000' };
    const meanY = pts.reduce((s, p) => s + p[1], 0) / n;
    let ssTot = 0, ssRes = 0;
    let dotXe = 0, sum_e = 0;
    for (const p of pts) {
      const yPred = this.w[0] * p[0] + (this.fitIntercept ? this.w[1] : 0);
      const e = p[1] - yPred;
      ssTot += (p[1] - meanY) ** 2;
      ssRes += e * e;
      dotXe += p[0] * e;
      sum_e += e;
    }
    const r2 = ssTot > 1e-7 ? Math.max(0, 1 - ssRes / ssTot) : 1.0;
    const orthNorm = Math.hypot(dotXe, this.fitIntercept ? sum_e : 0) / n;
    return { r2: r2.toFixed(3), orthogonality: orthNorm.toFixed(4) };
  }

  scrambleWeights() {
    this.stopAutoGD();
    // Intentionally set an inverted slope and large bias to create clearly visible errors and gradient
    this.w = [-0.65, 2.6];
    this.iterationCount = 0;
    this.statusType = "scrambled";
    this.statusMessage = "🎲 <b>Веса сбиты!</b> Линия намеренно удалена от оптимума. Ошибки невязок (красные пунктиры) велики, градиент ∇L ≠ 0. Нажмите <b>«Шаг GD»</b> или <b>«▶ Авто-спуск»</b>!";
    this.render();
    this.updateMathCard();
    this.updateStatusUI();
  }

  computeGradient() {
    const [w1, w0] = this.w;
    let grad_w1 = 0;
    let grad_w0 = 0;
    const n = this.points.length;

    for (const p of this.points) {
      const y_hat = w1 * p[0] + w0;
      const error = y_hat - p[1]; // y_hat - y
      grad_w1 += (2 / n) * error * p[0];
      grad_w0 += (2 / n) * error;
    }

    const norm = Math.hypot(grad_w1, grad_w0);
    return { grad_w1, grad_w0, norm };
  }

  runGradientDescentStep() {
    const { grad_w1, grad_w0, norm } = this.computeGradient();
    const alpha = 0.05;

    if (norm < 0.03) {
      this.statusType = "optimum";
      this.statusMessage = "🎯 <b>Модель уже в оптимуме!</b> Антиградиент -∇L ≈ 0 (шаг сдвига весов меньше 0.001). Нажмите <b>«🎲 Сбить веса»</b>, чтобы запустить спуск заново!";
      this.stopAutoGD();
      this.updateStatusUI();
      return false;
    }

    this.w[0] -= alpha * grad_w1;
    if (this.fitIntercept) this.w[1] -= alpha * grad_w0;
    this.iterationCount++;
    this.statusType = "stepping";
    this.statusMessage = `<b>Шаг GD #${this.iterationCount}:</b> Градиент [∇w₁=${grad_w1.toFixed(3)}, ∇w₀=${grad_w0.toFixed(3)}]. Веса сдвинуты на Δw = [${(-alpha*grad_w1).toFixed(3)}, ${(-alpha*grad_w0).toFixed(3)}]. Прямая поворачивается к точкам!`;

    this.render();
    this.updateMathCard();
    this.updateStatusUI();
    return true;
  }

  runAutoGD() {
    if (this.autoGDTimer) {
      this.stopAutoGD();
      return;
    }

    const { norm } = this.computeGradient();
    if (norm < 0.03) {
      this.scrambleWeights();
    }

    this.statusType = "running";
    const btn = document.getElementById('btn-auto-gd');
    if (btn) btn.textContent = '⏸ Пауза авто-спуска';

    this.autoGDTimer = setInterval(() => {
      const hasStep = this.runGradientDescentStep();
      if (!hasStep || this.iterationCount >= 80) {
        this.stopAutoGD();
        this.statusType = "optimum";
        this.statusMessage = `🎯 <b>Авто-спуск завершен за ${this.iterationCount} шагов!</b> Модель сошлась к оптимуму (градиент ∇L ≈ 0, невязки e ортогональны X).`;
        this.updateStatusUI();
      }
    }, 60);
  }

  stopAutoGD() {
    if (this.autoGDTimer) {
      clearInterval(this.autoGDTimer);
      this.autoGDTimer = null;
    }
    const btn = document.getElementById('btn-auto-gd');
    if (btn) btn.textContent = '▶ Авто-спуск GD';
  }

  updateStatusUI() {
    const statusEl = document.getElementById('ls-status-feedback');
    if (!statusEl) return;

    const { grad_w1, grad_w0, norm } = this.computeGradient();
    let badgeClass = "badge-green";
    let badgeText = "В оптимуме (∇L ≈ 0)";

    if (this.statusType === "scrambled") {
      badgeClass = "badge-red";
      badgeText = "Не в оптимуме (∇L высокий)";
    } else if (this.statusType === "stepping" || this.statusType === "running") {
      badgeClass = "badge-cyan";
      badgeText = `Идет GD (Шаг #${this.iterationCount})`;
    }

    statusEl.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.4rem; margin-bottom:0.5rem;">
        <span class="stat-badge ${badgeClass}">${badgeText}</span>
        <span style="font-family:var(--font-mono); font-size:0.8rem; color:var(--text-dim); font-weight:700;">
          ||∇L|| = ${norm.toFixed(3)}
        </span>
      </div>
      <div style="font-family:var(--font-mono); font-size:0.82rem; color:var(--text-main); margin-bottom:0.4rem; display:flex; gap:0.8rem;">
        <span>∂L/∂w₁ = <b>${grad_w1.toFixed(3)}</b></span>
        <span>∂L/∂w₀ = <b>${grad_w0.toFixed(3)}</b></span>
      </div>
      <div style="font-size:0.82rem; color:var(--text-muted); line-height:1.5;">
        ${this.statusMessage}
      </div>
    `;
  }

  render() {
    this.canvas.clear();
    this.canvas.drawGrid();

    const [slope, intercept] = this.w;
    const ctx = this.canvas.ctx;
    const pal = this.getPalette();

    // 1. Draw regression line across full canvas viewport
    const pTL = this.canvas.canvasToMath(0, 0);
    const pBR = this.canvas.canvasToMath(this.canvas.width, this.canvas.height);
    const xMin = Math.min(pTL.x, pBR.x) - 2;
    const xMax = Math.max(pTL.x, pBR.x) + 2;
    this.canvas.drawPolygon([
      [xMin, slope * xMin + intercept],
      [xMax, slope * xMax + intercept]
    ], null, pal.primary, 3.5);

    // 2. Draw residual error drop-lines e_i
    let totalResidualSq = 0;
    for (const p of this.points) {
      const y_hat = slope * p[0] + intercept;
      const residual = p[1] - y_hat;
      totalResidualSq += residual * residual;

      const pActual = this.canvas.mathToCanvas(p[0], p[1]);
      const pPred = this.canvas.mathToCanvas(p[0], y_hat);

      ctx.save();
      ctx.strokeStyle = pal.secondary;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(pActual.px, pActual.py);
      ctx.lineTo(pPred.px, pPred.py);
      ctx.stroke();

      // Small projection foot tick
      ctx.fillStyle = pal.secondary;
      ctx.beginPath();
      ctx.arc(pPred.px, pPred.py, 3.5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.restore();
    }

    this.canvas.drawHandles();
  }

  updateMathCard() {
    const [w1, w0] = this.w;
    const sign = w0 >= 0 ? '+' : '-';
    const katexEl = document.getElementById('m3-latex-model');
    if (katexEl) {
      if (window.katex) {
        const latex = `w = (X^T X)^{-1} X^T y \\implies \\hat{y} = ${w1.toFixed(2)} x ${sign} ${Math.abs(w0).toFixed(2)}`;
        window.katex.render(latex, katexEl, { throwOnError: false });
      } else {
        katexEl.innerHTML = `<span style="font-family:var(--font-mono); color:#0284c7;">w = (XᵀX)⁻¹Xᵀy => ŷ = ${w1.toFixed(2)} x ${sign} ${Math.abs(w0).toFixed(2)}</span>`;
      }
    }

    let ss_res = 0;
    let meanY = 0;
    this.points.forEach(p => meanY += p[1]);
    meanY /= this.points.length;
    let ss_tot = 0;

    let x_dot_e = 0;
    let sum_e = 0;

    for (const p of this.points) {
      const y_hat = w1 * p[0] + w0;
      const residual = p[1] - y_hat;
      ss_res += residual ** 2;
      ss_tot += (p[1] - meanY) ** 2;
      x_dot_e += p[0] * residual;
      sum_e += residual;
    }
    const r2 = ss_tot > 1e-8 ? 1.0 - ss_res / ss_tot : 1.0;
    const mse = ss_res / this.points.length;
    const orthNorm = Math.hypot(x_dot_e, sum_e);

    let orthBadge = 'badge-green';
    let orthText = 'Ортогональность: Xᵀ e ≈ 0 (Оптимум)';
    if (orthNorm > 0.15) {
      orthBadge = 'badge-amber';
      orthText = `Не ортогонально: ||Xᵀ e|| = ${orthNorm.toFixed(2)}`;
    }

    const statsEl = document.getElementById('m3-stats-info');
    if (statsEl) {
      statsEl.innerHTML = `
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.4rem;">
          <span class="stat-badge badge-green">R² = ${r2.toFixed(3)}</span>
          <span class="stat-badge badge-amber">MSE = ${mse.toFixed(3)}</span>
          <span class="stat-badge ${orthBadge}">${orthText}</span>
        </div>
        <p style="font-size:0.82rem; color:var(--text-muted); line-height:1.5;">
          Связь алгебры и ML: вектор невязок <b>e = y - ŷ</b> ортогонален признакам <b>X</b> (<i>Xᵀ e = 0</i>) в точности тогда, когда градиент функции потерь равен нулю (<i>∇MSE = -²⁄ₙ Xᵀ e = 0</i>).
        </p>
      `;
    }
  }
}


// ============================================================
// 6. MODULE 4: INTERACTIVE TRAINER (PRACTICUM)
// ============================================================
class ModuleTrainer {
  constructor(canvas) {
    this.canvas = canvas;
    this.currentChallenge = null;
    this.userMatrix = [[1.0, 0.0], [0.0, 1.0]];
    this.userAngle = 0.0;
    this.userRank = 1;
    this.userSVM = { w1: 1.0, w2: 1.0, b: 0.0 };
    this.userLS = { slope: 0.8, intercept: 0.0 };
    this.score = 0;
    this.completedTasks = {};
    this.paletteId = 1;

    this.initCanvasHandles();
  }

  setPalette(pId, mode = 'light') {
    this.paletteId = parseInt(pId, 10) || 1;
    this.modeTheme = mode;
    this.render();
  }

  getPalette() {
    return window.getAppPalette(this.paletteId, (this.canvas ? this.canvas.mode : this.modeTheme) || 'light');
  }

  initCanvasHandles() {
    this.canvas.clearHandles();
    this.canvas.onHandleDrag = (id, x, y) => {
      if (id === 'train_i') {
        this.userMatrix[0][0] = Math.round(x * 10) / 10;
        this.userMatrix[1][0] = Math.round(y * 10) / 10;
        this.syncInputs();
        this.render();
      } else if (id === 'train_j') {
        this.userMatrix[0][1] = Math.round(x * 10) / 10;
        this.userMatrix[1][1] = Math.round(y * 10) / 10;
        this.syncInputs();
        this.render();
      } else if (id === 'train_rag_q' || id === 'train_eig_v') {
        let deg = Math.round((Math.atan2(y, x) * 180) / Math.PI);
        if (deg < 0) deg += 360;
        this.userAngle = deg;
        const slider = document.getElementById('tr-angle-slider');
        const valEl = document.getElementById('tr-angle-val');
        if (slider) slider.value = deg;
        if (valEl) valEl.textContent = `${deg}°`;
        this.render();
      } else if (id === 'train_svm_w') {
        this.userSVM.w1 = Math.round(x * 10) / 10;
        this.userSVM.w2 = Math.round(y * 10) / 10;
        this.syncInputs();
        this.render();
      }
    };
    this.canvas.onPan = () => this.render();
    this.canvas.onZoom = () => this.render();
  }

  syncInputs() {
    const el = (id) => document.getElementById(id);
    if (el('tr_m11')) el('tr_m11').value = this.userMatrix[0][0];
    if (el('tr_m12')) el('tr_m12').value = this.userMatrix[0][1];
    if (el('tr_m21')) el('tr_m21').value = this.userMatrix[1][0];
    if (el('tr_m22')) el('tr_m22').value = this.userMatrix[1][1];
    if (el('tr_w1')) el('tr_w1').value = this.userSVM.w1;
    if (el('tr_w2')) el('tr_w2').value = this.userSVM.w2;
    if (el('tr_b')) el('tr_b').value = this.userSVM.b;
    if (el('tr_slope')) el('tr_slope').value = this.userLS.slope;
    if (el('tr_intercept')) el('tr_intercept').value = this.userLS.intercept;
  }

  async loadChallenge(type = 'vector_mapping') {
    try {
      const res = await fetch(`/api/challenges/generate/${type}`);
      if (!res.ok) throw new Error("Status " + res.status);
      this.currentChallenge = await res.json();
    } catch (err) {
      this.currentChallenge = this.generateLocalChallenge(type);
    }
    this.resetUserInputs();
    this.render();
    this.updateChallengeUI();
  }

  generateLocalChallenge(type) {
    if (type === 'vector_mapping') {
      return {
        id: "vector_mapping",
        title: "Подбор матрицы перехода",
        instruction: "Подберите матрицу W, переводящую u = [1, 1] в v = [2, -1].",
        u: [1, 1],
        target_v: [2, -1],
        hint: "Вектор результата W u = [w11 + w12, w21 + w22]. Подберите такие коэффициенты, чтобы получить [2, -1]."
      };
    } else if (type === 'matrix_inversion') {
      return {
        id: "matrix_inversion",
        title: "Обращение деформации",
        instruction: "Дана матрица сдвига W = [[1, 1], [0, 1]]. Найдите матрицу W_inv, чтобы восстановить исходную сетку.",
        W: [[1, 1], [0, 1]],
        hint: "Формула 2x2: inv(W) = 1/det * [[d, -b], [-c, a]]. Обратная матрица к сдвигу вправо — сдвиг влево W_inv = [[1, -1], [0, 1]].",
        target_inverse: [[1, -1], [0, 1]]
      };
    } else if (type === 'svd_energy') {
      return {
        id: "svd_energy",
        title: "Энергия сингулярного разложения (SVD)",
        instruction: "Даны сингулярные числа матрицы: σ = [8.0, 3.0, 1.0]. Определите минимальный ранг k, чтобы сохранить не менее 90% энергии матрицы (сумма квадратов σᵢ).",
        singular_values: [8.0, 3.0, 1.0],
        total_energy: 74.0,
        correct_rank: 2,
        cumulative_energies: [86.5, 98.6, 100.0],
        hint: "Формула энергии: E(k) = (σ₁² + ... + σₖ²) / (Σ σᵢ²). При k=1: 64/74 ≈ 86.5%, при k=2: (64+9)/74 ≈ 98.6% >= 90%."
      };
    } else if (type === 'rag_cosine') {
      return {
        id: "rag_cosine",
        title: "Векторный таргетинг RAG (Косинусное сходство)",
        instruction: "Направьте поисковый запрос (угол θ) так, чтобы косинусное сходство с целевым документом составило не менее 0.95.",
        target_vector: [1.77, 1.77],
        target_angle_deg: 45,
        hint: "Косинусное сходство cos(θ) >= 0.95 достигается, когда угол отклонения от вектора цели меньше 18°."
      };
    } else if (type === 'pca_variance') {
      const points = [];
      const rad = 0.7;
      for (let i = 0; i < 30; i++) {
        const u = (Math.random() - 0.5) * 5;
        const v = (Math.random() - 0.5) * 1.2;
        points.push([u * Math.cos(rad) - v * Math.sin(rad), u * Math.sin(rad) + v * Math.cos(rad)]);
      }
      return {
        id: "pca_variance",
        title: "Охота за главной осью дисперсии",
        instruction: "Задайте угол theta (в градусах) оси проекции, чтобы максимизировать сохраняемую дисперсию выборки.",
        points,
        true_angle_deg: 40.0,
        hint: "Главная компонента направлена вдоль наибольшей вытянутости облака точек (~40°)."
      };
    } else if (type === 'class_separation') {
      const points = [
        { x: 1.2, y: 1.8, label: 1 }, { x: 2.0, y: 1.0, label: 1 }, { x: 1.5, y: 2.5, label: 1 },
        { x: 2.8, y: 2.0, label: 1 }, { x: 0.8, y: 2.8, label: 1 }, { x: 3.0, y: 1.2, label: 1 },
        { x: -1.2, y: -0.8, label: -1 }, { x: -2.0, y: 0.2, label: -1 }, { x: -0.5, y: -2.0, label: -1 },
        { x: -1.8, y: -1.8, label: -1 }, { x: -2.8, y: -0.5, label: -1 }, { x: -1.0, y: -2.8, label: -1 }
      ];
      return {
        id: "class_separation",
        title: "Раздели классы (Гиперплоскость)",
        instruction: "Подберите веса w1, w2 и сдвиг b, чтобы гиперплоскость w1*x + w2*y + b = 0 разделила все синие (+1) и красные (-1) точки (Accuracy = 100%).",
        points,
        hint: "Вектор весов w = [w1, w2] задает нормаль, направленную в сторону синих точек (+1), а b сдвигает разделяющую прямую."
      };
    } else if (type === 'invariant_eigenvector') {
      return {
        id: "invariant_eigenvector",
        title: "Поиск инвариантного направления (Собственный вектор)",
        instruction: "Дана матрица оператора W = [[2, 1], [1, 2]]. Поверните вектор v (угол θ), чтобы W v был строго коллинеарен v (W v = λ v).",
        W: [[2, 1], [1, 2]],
        target_eigenvalues: [3.0, 1.0],
        target_angles_deg: [45.0, 135.0],
        hint: "Для матрицы [[2, 1], [1, 2]] собственные векторы направлены под углами 45° (λ=3) и 135° (λ=1)."
      };
    } else if (type === 'least_squares_fit') {
      const points = [
        [-3.0, -1.8], [-2.0, -0.9], [-1.0, 0.2], [0.0, 1.1],
        [1.0, 2.0], [2.0, 3.1], [3.0, 3.9]
      ];
      return {
        id: "least_squares_fit",
        title: "Подбор проекции МНК на глаз",
        instruction: "Подберите наклон k и сдвиг b прямой y = k*x + b, минимизируя длину вектора вертикальных невязок ||e||.",
        points,
        optimal_slope: 0.98,
        optimal_intercept: 1.08,
        optimal_residual_norm: 0.32,
        hint: "Выставьте прямую так, чтобы сумма квадратов вертикальных отклонений точек от прямой была минимальной."
      };
    } else {
      return this.generateLocalChallenge('vector_mapping');
    }
  }

  resetUserInputs() {
    this.userMatrix = [[1.0, 0.0], [0.0, 1.0]];
    this.userAngle = 0.0;
    this.userRank = 1;
    this.userSVM = { w1: 1.0, w2: 1.0, b: 0.0 };
    this.userLS = { slope: 0.5, intercept: 0.0 };
    this.canvas.clearHandles();
    const pal = this.getPalette();

    if (this.currentChallenge?.id === 'vector_mapping') {
      this.canvas.addHandle('train_i', 1, 0, 8, pal.primary, 'î');
      this.canvas.addHandle('train_j', 0, 1, 8, pal.secondary, 'ĵ');
    } else if (this.currentChallenge?.id === 'rag_cosine') {
      const rad = (this.userAngle * Math.PI) / 180;
      this.canvas.addHandle('train_rag_q', Math.cos(rad) * 2.5, Math.sin(rad) * 2.5, 9, pal.secondary, 'q');
    } else if (this.currentChallenge?.id === 'class_separation') {
      this.canvas.addHandle('train_svm_w', 1.0, 1.0, 8.5, pal.primary, 'w');
    } else if (this.currentChallenge?.id === 'invariant_eigenvector') {
      const rad = (this.userAngle * Math.PI) / 180;
      this.canvas.addHandle('train_eig_v', Math.cos(rad) * 2.5, Math.sin(rad) * 2.5, 8.5, pal.primary, 'v');
    }
    this.syncInputs();
    const rSlider = document.getElementById('tr-rank-slider');
    const rVal = document.getElementById('tr-rank-val');
    if (rSlider) rSlider.value = 1;
    if (rVal) rVal.textContent = 'k = 1';
    const aSlider = document.getElementById('tr-angle-slider');
    const aVal = document.getElementById('tr-angle-val');
    if (aSlider) aSlider.value = 0;
    if (aVal) aVal.textContent = '0°';
  }

  updateChallengeUI() {
    const ch = this.currentChallenge;
    if (!ch) return;

    document.getElementById('challenge-title').textContent = ch.title;
    document.getElementById('challenge-instruction').textContent = ch.instruction;
    document.getElementById('challenge-hint').textContent = `💡 Подсказка: ${ch.hint}`;
    document.getElementById('challenge-feedback').innerHTML = '';

    const matrixBox = document.getElementById('challenge-matrix-box');
    const angleBox = document.getElementById('challenge-angle-box');
    const rankBox = document.getElementById('challenge-rank-box');
    const svmBox = document.getElementById('challenge-svm-box');
    const lsBox = document.getElementById('challenge-ls-box');

    if (matrixBox) matrixBox.style.display = 'none';
    if (angleBox) angleBox.style.display = 'none';
    if (rankBox) rankBox.style.display = 'none';
    if (svmBox) svmBox.style.display = 'none';
    if (lsBox) lsBox.style.display = 'none';

    if (ch.id === 'pca_variance' || ch.id === 'rag_cosine' || ch.id === 'invariant_eigenvector') {
      if (angleBox) {
        angleBox.style.display = 'block';
        const label = angleBox.querySelector('.slider-label span:first-child');
        if (label) {
          label.textContent = ch.id === 'pca_variance' ? 'Угол поворота главной оси θ:' :
                              ch.id === 'invariant_eigenvector' ? 'Угол поворота вектора v (θ):' :
                              'Угол вектора поискового запроса θ:';
        }
      }
    } else if (ch.id === 'svd_energy') {
      if (rankBox) {
        rankBox.style.display = 'block';
        const slider = document.getElementById('tr-rank-slider');
        if (slider && ch.singular_values) {
          slider.max = ch.singular_values.length;
          slider.value = 1;
        }
      }
    } else if (ch.id === 'class_separation') {
      if (svmBox) svmBox.style.display = 'block';
    } else if (ch.id === 'least_squares_fit') {
      if (lsBox) lsBox.style.display = 'block';
    } else {
      if (matrixBox) matrixBox.style.display = 'block';
    }
  }

  render() {
    this.canvas.clear();
    this.canvas.drawGrid();

    const ch = this.currentChallenge;
    if (!ch) return;

    const ctx = this.canvas.ctx;
    const pal = this.getPalette();
    const isDark = this.canvas && this.canvas.mode === 'dark';

    if (ch.id === 'vector_mapping') {
      const u = ch.u;
      // Draw initial vector u
      this.canvas.drawVector(0, 0, u[0], u[1], isDark ? '#94a3b8' : '#64748b', 'u (исходный)', 2.5);
      // Target vector v
      this.canvas.drawVector(0, 0, ch.target_v[0], ch.target_v[1], pal.accent, 'Цель (v)', 4);
      const resX = this.userMatrix[0][0] * u[0] + this.userMatrix[0][1] * u[1];
      const resY = this.userMatrix[1][0] * u[0] + this.userMatrix[1][1] * u[1];
      this.canvas.drawVector(0, 0, resX, resY, pal.primary, 'W·u', 3.5);
      this.canvas.drawVector(0, 0, this.userMatrix[0][0], this.userMatrix[1][0], pal.primary, 'î', 2);
      this.canvas.drawVector(0, 0, this.userMatrix[0][1], this.userMatrix[1][1], pal.secondary, 'ĵ', 2);
      this.canvas.drawHandles();
    } else if (ch.id === 'matrix_inversion') {
      // Deformed space W
      this.canvas.drawTransformedGrid(ch.W, [0, 0], 5);
      this.canvas.drawVector(0, 0, ch.W[0][0], ch.W[1][0], isDark ? '#64748b' : '#94a3b8', 'W·î', 2);
      this.canvas.drawVector(0, 0, ch.W[0][1], ch.W[1][1], isDark ? '#64748b' : '#94a3b8', 'W·ĵ', 2);

      // Product matrix: W_user * W
      const M = this.userMatrix;
      const W = ch.W;
      const p11 = M[0][0] * W[0][0] + M[0][1] * W[1][0];
      const p12 = M[0][0] * W[0][1] + M[0][1] * W[1][1];
      const p21 = M[1][0] * W[0][0] + M[1][1] * W[1][0];
      const p22 = M[1][0] * W[0][1] + M[1][1] * W[1][1];

      // Draw restored vectors under W_user * W (aiming for standard basis I = [1,0], [0,1])
      this.canvas.drawVector(0, 0, p11, p21, pal.primary, '(W⁻¹·W)·î', 3.5);
      this.canvas.drawVector(0, 0, p12, p22, pal.secondary, '(W⁻¹·W)·ĵ', 3.5);
    } else if (ch.id === 'pca_variance') {
      for (const p of ch.points) {
        const { px, py } = this.canvas.mathToCanvas(p[0], p[1]);
        ctx.save();
        ctx.beginPath();
        ctx.arc(px, py, 4.5, 0, 2 * Math.PI);
        ctx.fillStyle = pal.scatter;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = isDark ? '#f4f4f5' : '#000000';
        ctx.stroke();
        ctx.restore();
      }

      const rad = (this.userAngle * Math.PI) / 180;
      const dx = Math.cos(rad) * 6;
      const dy = Math.sin(rad) * 6;
      this.canvas.drawPolygon([[-dx, -dy], [dx, dy]], null, pal.secondary, 2.5, true);
      this.canvas.drawVector(0, 0, dx * 0.4, dy * 0.4, pal.secondary, 'Ось θ', 3.5);
    } else if (ch.id === 'svd_energy') {
      // Draw singular spectrum bar chart
      const sigmas = ch.singular_values || [8.0, 3.0, 1.0];
      const maxVal = Math.max(...sigmas, 1.0);
      const totalEnergy = ch.total_energy || sigmas.reduce((acc, v) => acc + v * v, 0);
      const w = this.canvas.displayWidth || this.canvas.width || 700;
      const h = this.canvas.displayHeight || this.canvas.height || 520;
      const startX = 60;
      const startY = h - 60;
      const chartW = w - 120;
      const chartH = h - 130;
      const barWidth = Math.min(65, (chartW / sigmas.length) - 20);

      ctx.save();
      ctx.strokeStyle = isDark ? '#3f3f46' : '#cbd5e1';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(startX - 10, startY);
      ctx.lineTo(startX + chartW + 10, startY);
      ctx.moveTo(startX, startY);
      ctx.lineTo(startX, startY - chartH - 20);
      ctx.stroke();

      let cumSq = 0;
      const energyPoints = [];
      for (let i = 0; i < sigmas.length; i++) {
        const s = sigmas[i];
        cumSq += s * s;
        const pct = ((cumSq / totalEnergy) * 100).toFixed(1);
        const x = startX + i * (chartW / sigmas.length) + 15;
        const barH = (s / maxVal) * chartH;
        const y = startY - barH;
        const isKept = i < this.userRank;

        ctx.fillStyle = isKept ? pal.primary : (isDark ? '#3f3f46' : '#e2e8f0');
        ctx.fillRect(x, y, barWidth, barH);
        ctx.strokeStyle = isDark ? '#f4f4f5' : '#000000';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, barWidth, barH);

        ctx.fillStyle = isDark ? '#f4f4f5' : '#09090b';
        ctx.font = '700 12px var(--font-heading)';
        ctx.textAlign = 'center';
        ctx.fillText(`σ${i + 1} = ${s}`, x + barWidth / 2, startY + 20);
        ctx.fillText(`Σ = ${pct}%`, x + barWidth / 2, y - 8);

        const energyY = startY - (cumSq / totalEnergy) * chartH;
        energyPoints.push({ x: x + barWidth / 2, y: energyY });
      }

      // Draw Cumulative Energy line (Pareto curve)
      if (energyPoints.length > 0) {
        ctx.strokeStyle = '#8b5cf6';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(energyPoints[0].x, energyPoints[0].y);
        for (let i = 1; i < energyPoints.length; i++) {
          ctx.lineTo(energyPoints[i].x, energyPoints[i].y);
        }
        ctx.stroke();

        for (const ep of energyPoints) {
          ctx.fillStyle = '#8b5cf6';
          ctx.beginPath();
          ctx.arc(ep.x, ep.y, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // 90% threshold line
      const y90 = startY - 0.9 * chartH;
      ctx.strokeStyle = '#ea580c';
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(startX, y90);
      ctx.lineTo(startX + chartW, y90);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#ea580c';
      ctx.font = '700 11px var(--font-mono)';
      ctx.textAlign = 'right';
      ctx.fillText('Порог 90% энергии (кривая Σ)', startX + chartW, y90 - 6);
      ctx.restore();
    } else if (ch.id === 'rag_cosine') {
      // Draw semantic sphere
      this.canvas.drawCircle(0, 0, 2.5, isDark ? 'rgba(56, 189, 248, 0.05)' : 'rgba(2, 132, 199, 0.05)', isDark ? '#334155' : '#cbd5e1', 1.5, true);

      // Target document vector
      const tv = ch.target_vector || [1.77, 1.77];
      this.canvas.drawVector(0, 0, tv[0], tv[1], pal.primary, 'Целевой документ (Target)', 4);

      // User query vector
      const rad = (this.userAngle * Math.PI) / 180;
      const qx = Math.cos(rad) * 2.5;
      const qy = Math.sin(rad) * 2.5;
      this.canvas.drawVector(0, 0, qx, qy, pal.secondary, 'Запрос (Query q)', 3.5);

      // Arc between target and query
      const targetRad = ((ch.target_angle_deg || 45) * Math.PI) / 180;
      this.canvas.drawAngleArc(0, 0, 1.3, targetRad, rad, pal.secondary);

      // Live Cosine value on canvas
      const diffRad = Math.abs(rad - targetRad);
      const cosVal = Math.cos(diffRad);
      ctx.save();
      ctx.font = '800 14px var(--font-mono)';
      ctx.fillStyle = cosVal >= 0.95 ? '#16a34a' : (isDark ? '#f4f4f5' : '#000000');
      ctx.fillText(`cos(θ) = ${cosVal.toFixed(4)} ${cosVal >= 0.95 ? '✓ (Цель достигнута!)' : ''}`, 20, 30);
      ctx.restore();

      const hObj = this.canvas.handles.find(h => h.id === 'train_rag_q');
      if (hObj) {
        hObj.x = qx;
        hObj.y = qy;
      }
      this.canvas.drawHandles();
    } else if (ch.id === 'class_separation') {
      const w1 = this.userSVM.w1, w2 = this.userSVM.w2, b = this.userSVM.b;
      const normW = Math.hypot(w1, w2) || 1e-6;

      const span = 8;
      let p1, p2;
      if (Math.abs(w2) > Math.abs(w1)) {
        p1 = this.canvas.mathToCanvas(-span, (-w1 * -span - b) / w2);
        p2 = this.canvas.mathToCanvas(span, (-w1 * span - b) / w2);
      } else {
        p1 = this.canvas.mathToCanvas((-w2 * -span - b) / w1, -span);
        p2 = this.canvas.mathToCanvas((-w2 * span - b) / w1, span);
      }
      ctx.save();
      ctx.strokeStyle = isDark ? '#ffffff' : '#000000';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(p1.px, p1.py);
      ctx.lineTo(p2.px, p2.py);
      ctx.stroke();
      ctx.restore();

      const x0 = -b * w1 / (normW * normW);
      const y0 = -b * w2 / (normW * normW);
      this.canvas.drawVector(x0, y0, x0 + (w1 / normW) * 1.6, y0 + (w2 / normW) * 1.6, pal.primary, 'w', 3);

      let correct = 0;
      let minMargin = 999;
      for (const pt of ch.points) {
        const { px, py } = this.canvas.mathToCanvas(pt.x, pt.y);
        const score = w1 * pt.x + w2 * pt.y + b;
        const isRight = (score >= 0 && pt.label === 1) || (score < 0 && pt.label === -1);
        if (isRight) correct++;
        const dist = (pt.label * score) / normW;
        if (dist < minMargin) minMargin = dist;

        ctx.save();
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, 2 * Math.PI);
        ctx.fillStyle = pt.label === 1 ? pal.primary : pal.secondary;
        ctx.fill();
        ctx.strokeStyle = isRight ? (isDark ? '#f4f4f5' : '#000000') : '#ef4444';
        ctx.lineWidth = isRight ? 1.5 : 2.5;
        ctx.stroke();
        ctx.restore();
      }

      const acc = Math.round((correct / ch.points.length) * 100);
      ctx.save();
      ctx.font = '800 13px JetBrains Mono, monospace';
      ctx.fillStyle = acc === 100 ? '#16a34a' : (isDark ? '#f4f4f5' : '#000000');
      ctx.fillText(`Точность: ${acc}% (${correct}/${ch.points.length}) ${acc === 100 ? '✓ Отлично!' : ''} | Margin: ${Math.max(0, minMargin).toFixed(2)}`, 20, 30);
      ctx.restore();

      const hObj = this.canvas.handles.find(h => h.id === 'train_svm_w');
      if (hObj) {
        hObj.x = w1;
        hObj.y = w2;
      }
      this.canvas.drawHandles();
    } else if (ch.id === 'invariant_eigenvector') {
      const W = ch.W || [[2, 1], [1, 2]];
      const rad = (this.userAngle * Math.PI) / 180;
      const vx = Math.cos(rad) * 2.5;
      const vy = Math.sin(rad) * 2.5;
      const Wvx = W[0][0] * vx + W[0][1] * vy;
      const Wvy = W[1][0] * vx + W[1][1] * vy;

      const normV = Math.hypot(vx, vy) || 1;
      const normWv = Math.hypot(Wvx, Wvy) || 1;
      const sinDiff = Math.abs(vx * Wvy - vy * Wvx) / (normV * normWv);
      const isCollinear = sinDiff < 0.08;
      const lambda = ((vx * Wvx + vy * Wvy) / (normV * normV)).toFixed(2);

      this.canvas.drawVector(0, 0, vx, vy, pal.primary, 'v', 3.5);
      this.canvas.drawVector(0, 0, Wvx, Wvy, pal.secondary, `W·v (${isCollinear ? 'λ=' + lambda : ''})`, 3.5);

      ctx.save();
      ctx.font = '800 13px JetBrains Mono, monospace';
      ctx.fillStyle = isCollinear ? '#16a34a' : (isDark ? '#f4f4f5' : '#000000');
      ctx.fillText(`Коллинеарность: ${isCollinear ? '✓ ДА (Собственный вектор!)' : 'НЕТ'} | Ошибка: ${sinDiff.toFixed(3)}`, 20, 30);
      ctx.restore();

      const hObj = this.canvas.handles.find(h => h.id === 'train_eig_v');
      if (hObj) {
        hObj.x = vx;
        hObj.y = vy;
      }
      this.canvas.drawHandles();
    } else if (ch.id === 'least_squares_fit') {
      const pts = ch.points || [];
      const slope = this.userLS.slope;
      const intercept = this.userLS.intercept;

      for (const p of pts) {
        const { px, py } = this.canvas.mathToCanvas(p[0], p[1]);
        ctx.save();
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, 2 * Math.PI);
        ctx.fillStyle = pal.primary;
        ctx.fill();
        ctx.strokeStyle = isDark ? '#ffffff' : '#000000';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        const yPred = slope * p[0] + intercept;
        const pPred = this.canvas.mathToCanvas(p[0], yPred);
        ctx.strokeStyle = '#ea580c';
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(pPred.px, pPred.py);
        ctx.stroke();
        ctx.restore();
      }

      const span = 8;
      const pL = this.canvas.mathToCanvas(-span, slope * -span + intercept);
      const pR = this.canvas.mathToCanvas(span, slope * span + intercept);
      ctx.save();
      ctx.strokeStyle = pal.secondary;
      ctx.lineWidth = 3.0;
      ctx.beginPath();
      ctx.moveTo(pL.px, pL.py);
      ctx.lineTo(pR.px, pR.py);
      ctx.stroke();
      ctx.restore();

      let sumSq = 0;
      for (const p of pts) {
        sumSq += (p[1] - (slope * p[0] + intercept)) ** 2;
      }
      const userNorm = Math.sqrt(sumSq);
      const optNorm = ch.optimal_residual_norm || 0.32;
      const ratio = userNorm / optNorm;
      const isGood = ratio <= 1.25;

      ctx.save();
      ctx.font = '800 13px JetBrains Mono, monospace';
      ctx.fillStyle = isGood ? '#16a34a' : (isDark ? '#f4f4f5' : '#000000');
      ctx.fillText(`||e|| = ${userNorm.toFixed(2)} | Оптимум МНК = ${optNorm.toFixed(2)} ${isGood ? '✓ (Отлично!)' : ''}`, 20, 30);
      ctx.restore();
    }
  }

  async verifySubmission() {
    if (!this.currentChallenge) return;
    const ch = this.currentChallenge;

    let payload = {};
    if (ch.id === 'vector_mapping') {
      payload = { u: ch.u, target_v: ch.target_v, matrix: this.userMatrix };
    } else if (ch.id === 'matrix_inversion') {
      payload = { W: ch.W, matrix: this.userMatrix };
    } else if (ch.id === 'pca_variance') {
      payload = { angle_deg: this.userAngle, true_angle_deg: ch.true_angle_deg };
    } else if (ch.id === 'svd_energy') {
      payload = { rank: this.userRank, correct_rank: ch.correct_rank, cumulative_energies: ch.cumulative_energies };
    } else if (ch.id === 'rag_cosine') {
      payload = { angle_deg: this.userAngle, target_angle_deg: ch.target_angle_deg };
    } else if (ch.id === 'class_separation') {
      payload = { points: ch.points, w1: this.userSVM.w1, w2: this.userSVM.w2, b: this.userSVM.b };
    } else if (ch.id === 'invariant_eigenvector') {
      payload = { W: ch.W, angle_deg: this.userAngle };
    } else if (ch.id === 'least_squares_fit') {
      payload = { points: ch.points, slope: this.userLS.slope, intercept: this.userLS.intercept, optimal_residual_norm: ch.optimal_residual_norm };
    }

    let data;
    try {
      const res = await fetch(`/api/challenges/verify/${ch.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ submission: payload })
      });
      if (!res.ok) throw new Error("Status " + res.status);
      data = await res.json();
    } catch (err) {
      data = this.verifyLocalChallenge(ch.id, payload);
    }

    const fb = document.getElementById('challenge-feedback');
    if (data.success) {
      if (!this.completedTasks[ch.id]) {
        this.completedTasks[ch.id] = true;
        this.score += 100;
        document.getElementById('user-score').textContent = this.score;
      }
      fb.innerHTML = `
        <div class="toast success" style="position:static; margin-top:0.75rem;">
          🎉 ${data.message} (+100 очков)
        </div>
      `;
    } else {
      fb.innerHTML = `
        <div class="toast warn" style="position:static; margin-top:0.75rem;">
          ⚠️ ${data.message}
        </div>
      `;
    }
  }

  verifyLocalChallenge(id, payload) {
    if (id === 'vector_mapping') {
      const u = payload.u;
      const target_v = payload.target_v;
      const M = payload.matrix;
      const rx = M[0][0] * u[0] + M[0][1] * u[1];
      const ry = M[1][0] * u[0] + M[1][1] * u[1];
      const err = Math.hypot(rx - target_v[0], ry - target_v[1]);
      const ok = err < 0.2;
      return {
        success: ok,
        message: ok ? "Отлично! Вектор точно переведен в цель!" : `Погрешность: ${err.toFixed(2)}. Получился [${rx.toFixed(2)}, ${ry.toFixed(2)}].`
      };
    } else if (id === 'matrix_inversion') {
      const W = payload.W;
      const M = payload.matrix;
      const p11 = M[0][0] * W[0][0] + M[0][1] * W[1][0];
      const p12 = M[0][0] * W[0][1] + M[0][1] * W[1][1];
      const p21 = M[1][0] * W[0][0] + M[1][1] * W[1][0];
      const p22 = M[1][0] * W[0][1] + M[1][1] * W[1][1];
      const err = Math.hypot(p11 - 1, p12, p21, p22 - 1);
      const ok = err < 0.25;
      return {
        success: ok,
        message: ok ? "Верно! Произведение матриц восстановило единичный базис I." : `Матрица не является обратной. Ошибка нормы: ${err.toFixed(2)}.`
      };
    } else if (id === 'svd_energy') {
      const ok = payload.rank === payload.correct_rank;
      const pct = payload.cumulative_energies ? payload.cumulative_energies[payload.rank - 1] : 0;
      return {
        success: ok,
        message: ok ? `Верно! При k=${payload.rank} сохраняется ${pct}% энергии (>= 90%).` : `Не совсем. При k=${payload.rank} сохраняется ${pct}% энергии. Правильный ранг: k=${payload.correct_rank}.`
      };
    } else if (id === 'rag_cosine') {
      const diffRad = Math.abs(payload.angle_deg - payload.target_angle_deg) * Math.PI / 180;
      const cosSim = Math.cos(diffRad);
      const ok = cosSim >= 0.95;
      return {
        success: ok,
        message: ok ? `Отличный таргетинг! Косинусное сходство: ${cosSim.toFixed(4)} >= 0.95.` : `Косинусное сходство: ${cosSim.toFixed(4)} < 0.95. Поверните запрос ближе к цели.`
      };
    } else if (id === 'class_separation') {
      const pts = payload.points || [];
      const w1 = payload.w1, w2 = payload.w2, b = payload.b;
      let correct = 0;
      for (const p of pts) {
        const score = w1 * p.x + w2 * p.y + b;
        if ((score >= 0 && p.label === 1) || (score < 0 && p.label === -1)) correct++;
      }
      const ok = (correct === pts.length);
      return {
        success: ok,
        message: ok ? `Блестяще! Все ${pts.length} точек разделены со 100% точностью!` : `Точность: ${Math.round(correct/pts.length*100)}% (${correct}/${pts.length}). Подстройте нормаль w или сдвиг b.`
      };
    } else if (id === 'invariant_eigenvector') {
      const W = payload.W;
      const rad = (payload.angle_deg * Math.PI) / 180;
      const vx = Math.cos(rad) * 2.5, vy = Math.sin(rad) * 2.5;
      const Wvx = W[0][0] * vx + W[0][1] * vy;
      const Wvy = W[1][0] * vx + W[1][1] * vy;
      const normV = Math.hypot(vx, vy) || 1;
      const normWv = Math.hypot(Wvx, Wvy) || 1;
      const sinDiff = Math.abs(vx * Wvy - vy * Wvx) / (normV * normWv);
      const ok = sinDiff < 0.08;
      const lambda = ((vx * Wvx + vy * Wvy) / (normV * normV)).toFixed(2);
      return {
        success: ok,
        message: ok ? `Великолепно! Вектор v инвариантен: W v = ${lambda} v (λ = ${lambda})!` : `Вектор W v отклоняется от v (ошибка: ${sinDiff.toFixed(3)}). Поверните ближе.`
      };
    } else if (id === 'least_squares_fit') {
      const pts = payload.points || [];
      const slope = payload.slope, intercept = payload.intercept;
      let sumSq = 0;
      for (const p of pts) {
        sumSq += (p[1] - (slope * p[0] + intercept)) ** 2;
      }
      const userNorm = Math.sqrt(sumSq);
      const optNorm = payload.optimal_residual_norm || 0.32;
      const ratio = userNorm / optNorm;
      const ok = ratio <= 1.25;
      return {
        success: ok,
        message: ok ? `Отличный глазомер! Норма невязок ||e|| = ${userNorm.toFixed(2)} практически совпадает с минимумом МНК ${optNorm.toFixed(2)}!` : `Норма невязок ||e|| = ${userNorm.toFixed(2)} превышает оптимум МНК (${optNorm.toFixed(2)}). Подкорректируйте прямую.`
      };
    } else {
      const uAngle = payload.angle_deg % 180;
      const tAngle = (payload.true_angle_deg || 40) % 180;
      const diff = Math.min(Math.abs(uAngle - tAngle), 180 - Math.abs(uAngle - tAngle));
      const ok = diff <= 12;
      return {
        success: ok,
        message: ok ? `Блестяще! Вы нашли направление максимальной дисперсии с точностью до ${diff.toFixed(1)}°.` : `Отклонение от главной оси: ${diff.toFixed(1)}°. Попробуйте повернуть ближе.`
      };
    }
  }
}

// ============================================================
// 7. MODULE: SINGULAR VALUE DECOMPOSITION (SVD) & LoRA
// ============================================================
class ModuleSVD {
  constructor(canvas) {
    this.canvas = canvas;
    this.mode = 'geom'; // 'geom' | 'lora'
    this.stage = 3.0;   // 0 to 3 for transformation stages
    this.A = [[1.5, 1.0], [0.5, 1.5]];
    this.U = [[1, 0], [0, 1]];
    this.S = [1, 1];
    this.Vt = [[1, 0], [0, 1]];
    this.rank = 1;      // for LoRA (1 to 4)
    this.paletteId = 1;
    this.modeTheme = 'light';
    this.currentPattern = 'cross';

    this.patterns = {
      cross: [
        [0.0, 0.0, 0.9, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.9, 0.9, 0.0, 0.0],
        [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        [0.0, 0.0, 0.9, 0.9, 0.0, 0.0],
        [0.0, 0.0, 0.9, 0.9, 0.0, 0.0]
      ],
      box: [
        [0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.9],
        [0.9, 0.1, 0.0, 0.0, 0.1, 0.9],
        [0.9, 0.1, 0.0, 0.0, 0.1, 0.9],
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.9],
        [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
      ],
      gradient: [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.6, 0.7, 0.8, 0.9, 1.0, 1.0]
      ]
    };

    this.origMatrix = this.patterns.cross;
    this.approxMatrix = null;
    this.loraMetrics = null;
    this.loraSingulars = [1, 1, 1, 1, 1, 1];

    this.compute2DSVD();
    this.computeLoRA();
  }

  setPalette(pId, mode = 'light') {
    this.paletteId = parseInt(pId, 10) || 1;
    this.modeTheme = mode;
    this.render();
    this.updateGrids();
  }

  getPalette() {
    return window.getAppPalette(this.paletteId, (this.canvas ? this.canvas.mode : this.modeTheme) || 'light');
  }

  compute2DSVD() {
    const a = this.A[0][0], b = this.A[0][1];
    const c = this.A[1][0], d = this.A[1][1];

    // M = A^T * A
    const e = a * a + c * c;
    const f = a * b + c * d;
    const g = b * b + d * d;

    const trace = e + g;
    const det = e * g - f * f;
    const disc = Math.max(0, trace * trace - 4 * det);
    const sqrtDisc = Math.sqrt(disc);

    const l1 = Math.max(0, (trace + sqrtDisc) / 2);
    const l2 = Math.max(0, (trace - sqrtDisc) / 2);

    const s1 = Math.sqrt(l1);
    const s2 = Math.sqrt(l2);

    let v1 = [1, 0];
    if (Math.abs(f) > 1e-7) {
      v1 = [f, l1 - e];
      const norm = Math.hypot(v1[0], v1[1]);
      v1 = [v1[0] / norm, v1[1] / norm];
    } else if (e < g) {
      v1 = [0, 1];
    }
    const v2 = [-v1[1], v1[0]];

    let u1 = [1, 0];
    if (s1 > 1e-7) {
      u1 = [(a * v1[0] + b * v1[1]) / s1, (c * v1[0] + d * v1[1]) / s1];
    }
    let u2 = [-u1[1], u1[0]];
    if (s2 > 1e-7) {
      const candU2 = [(a * v2[0] + b * v2[1]) / s2, (c * v2[0] + d * v2[1]) / s2];
      const dot = u2[0] * candU2[0] + u2[1] * candU2[1];
      u2 = dot < 0 ? [-candU2[0], -candU2[1]] : candU2;
    }

    const a11_rec = s1 * u1[0] * v1[0] + s2 * u2[0] * v2[0];
    if (Math.abs(s1 * u1[0] * v1[0] - s2 * u2[0] * v2[0] - a) < Math.abs(a11_rec - a) - 1e-5) {
      u2 = [-u2[0], -u2[1]];
    }

    this.U = [[u1[0], u2[0]], [u1[1], u2[1]]];
    this.S = [s1, s2];
    this.Vt = [[v1[0], v1[1]], [v2[0], v2[1]]];
  }

  getInterpolatedMatrix() {
    const s = this.stage;
    const s1 = this.S[0];
    const s2 = this.S[1];
    const Vt = this.Vt;
    const U = this.U;

    if (s <= 1.0) {
      const t = Math.max(0, Math.min(1, s));
      return [
        [(1 - t) + t * Vt[0][0], t * Vt[0][1]],
        [t * Vt[1][0], (1 - t) + t * Vt[1][1]]
      ];
    } else if (s <= 2.0) {
      const t = s - 1.0;
      const curS1 = 1 + t * (s1 - 1);
      const curS2 = 1 + t * (s2 - 1);
      return [
        [curS1 * Vt[0][0], curS1 * Vt[0][1]],
        [curS2 * Vt[1][0], curS2 * Vt[1][1]]
      ];
    } else {
      const t = Math.min(1, s - 2.0);
      const u11 = (1 - t) + t * U[0][0];
      const u12 = t * U[0][1];
      const u21 = t * U[1][0];
      const u22 = (1 - t) + t * U[1][1];

      const sv11 = s1 * Vt[0][0], sv12 = s1 * Vt[0][1];
      const sv21 = s2 * Vt[1][0], sv22 = s2 * Vt[1][1];

      return [
        [u11 * sv11 + u12 * sv21, u11 * sv12 + u12 * sv22],
        [u21 * sv11 + u22 * sv21, u21 * sv12 + u22 * sv22]
      ];
    }
  }

  computeLoRA() {
    const A = this.origMatrix;
    const m = A.length;
    const n = A[0].length;

    // C = A * A^T
    const C = Array.from({ length: m }, () => Array(m).fill(0));
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) sum += A[i][k] * A[j][k];
        C[i][j] = sum;
      }
    }

    // Jacobi algorithm
    const U = Array.from({ length: m }, (_, i) => Array.from({ length: m }, (_, j) => i === j ? 1 : 0));
    for (let sweep = 0; sweep < 25; sweep++) {
      let maxOff = 0;
      for (let p = 0; p < m - 1; p++) {
        for (let q = p + 1; q < m; q++) {
          const off = Math.abs(C[p][q]);
          if (off > maxOff) maxOff = off;
          if (off > 1e-9) {
            const app = C[p][p], aqq = C[q][q], apq = C[p][q];
            const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
            const c = Math.cos(phi), s = Math.sin(phi);

            for (let i = 0; i < m; i++) {
              if (i !== p && i !== q) {
                const cip = C[i][p], ciq = C[i][q];
                C[i][p] = c * cip - s * ciq;
                C[p][i] = C[i][p];
                C[i][q] = s * cip + c * ciq;
                C[q][i] = C[i][q];
              }
            }
            C[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
            C[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
            C[p][q] = 0;
            C[q][p] = 0;

            for (let i = 0; i < m; i++) {
              const uip = U[i][p], uiq = U[i][q];
              U[i][p] = c * uip - s * uiq;
              U[i][q] = s * uip + c * uiq;
            }
          }
        }
      }
      if (maxOff < 1e-8) break;
    }

    const pairs = [];
    for (let i = 0; i < m; i++) {
      const val = Math.max(0, C[i][i]);
      const sigma = Math.sqrt(val);
      const uVec = U.map(row => row[i]);
      pairs.push({ sigma, u: uVec });
    }
    pairs.sort((a, b) => b.sigma - a.sigma);

    this.loraSingulars = pairs.map(p => p.sigma);
    const totalEnergy = this.loraSingulars.reduce((acc, s) => acc + s * s, 0);

    const k = Math.min(this.rank, m);
    const approx = Array.from({ length: m }, () => Array(n).fill(0));
    let kEnergy = 0;

    for (let r = 0; r < k; r++) {
      const { sigma, u } = pairs[r];
      kEnergy += sigma * sigma;
      if (sigma > 1e-7) {
        const v = [];
        for (let col = 0; col < n; col++) {
          let dot = 0;
          for (let row = 0; row < m; row++) dot += A[row][col] * u[row];
          v.push(dot / sigma);
        }
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < n; j++) {
            approx[i][j] += sigma * u[i] * v[j];
          }
        }
      }
    }

    let frobErrSq = 0;
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        frobErrSq += (A[i][j] - approx[i][j]) ** 2;
      }
    }
    const frobErr = Math.sqrt(frobErrSq);
    const energyPct = totalEnergy > 1e-9 ? (kEnergy / totalEnergy) * 100 : 100;
    const origParams = m * n;
    const loraParams = (m + n) * k;
    const compRatio = (origParams / loraParams);

    this.approxMatrix = approx;
    this.loraMetrics = {
      rank: k,
      energyPct: energyPct.toFixed(1),
      frobErr: frobErr.toFixed(3),
      origParams,
      loraParams,
      compRatio: compRatio.toFixed(1)
    };

    this.updateGrids();
  }

  setPreset(name) {
    if (name === 'stretch_rot') {
      this.A = [[1.5, 1.0], [0.5, 1.5]];
    } else if (name === 'shear') {
      this.A = [[1.0, 1.2], [0.0, 1.0]];
    } else if (name === 'singular') {
      this.A = [[1.2, 1.8], [0.8, 1.2]];
    } else if (name === 'orthogonal') {
      const rad = Math.PI / 4;
      this.A = [[Math.cos(rad), -Math.sin(rad)], [Math.sin(rad), Math.cos(rad)]];
    }
    this.syncInputs();
    this.compute2DSVD();
    this.render();
    this.updateMathCard();
  }

  setPattern(name) {
    if (this.patterns[name]) {
      this.currentPattern = name;
      this.origMatrix = this.patterns[name];
      this.computeLoRA();
      this.render();
    }
  }

  syncInputs() {
    const el = (id) => document.getElementById(id);
    if (el('svd-a11')) el('svd-a11').value = this.A[0][0].toFixed(1);
    if (el('svd-a12')) el('svd-a12').value = this.A[0][1].toFixed(1);
    if (el('svd-a21')) el('svd-a21').value = this.A[1][0].toFixed(1);
    if (el('svd-a22')) el('svd-a22').value = this.A[1][1].toFixed(1);
  }

  render() {
    this.canvas.clear();

    if (this.mode === 'geom') {
      this.renderGeometry();
    } else {
      this.renderLoRA();
    }
  }

  renderGeometry() {
    this.canvas.drawGrid();
    const pal = this.getPalette();
    const isDark = this.canvas && this.canvas.mode === 'dark';
    const ctx = this.canvas.ctx;
    const M = this.getInterpolatedMatrix();

    // 1. Draw transformed grid
    this.canvas.drawTransformedGrid(M, [0, 0], 4);

    // 2. Draw transformed unit circle
    const numPoints = 64;
    const circlePoints = [];
    for (let i = 0; i <= numPoints; i++) {
      const angle = (i / numPoints) * 2 * Math.PI;
      const x = Math.cos(angle);
      const y = Math.sin(angle);
      const tx = M[0][0] * x + M[0][1] * y;
      const ty = M[1][0] * x + M[1][1] * y;
      circlePoints.push([tx, ty]);
    }
    this.canvas.drawPolygon(circlePoints, isDark ? 'rgba(167, 139, 250, 0.15)' : 'rgba(139, 92, 246, 0.15)', pal.accent, 2.5, true);

    // 3. Right singular vectors v1, v2
    const v1 = [this.Vt[0][0], this.Vt[0][1]];
    const v2 = [this.Vt[1][0], this.Vt[1][1]];

    const tv1_x = M[0][0] * v1[0] + M[0][1] * v1[1];
    const tv1_y = M[1][0] * v1[0] + M[1][1] * v1[1];
    const tv2_x = M[0][0] * v2[0] + M[0][1] * v2[1];
    const tv2_y = M[1][0] * v2[0] + M[1][1] * v2[1];

    const s = this.stage;
    let lbl1 = 'M·v₁';
    let lbl2 = 'M·v₂';
    if (s <= 0.1) {
      lbl1 = 'v₁';
      lbl2 = 'v₂';
    } else if (s >= 2.9) {
      lbl1 = `σ₁·u₁ (${this.S[0].toFixed(2)})`;
      lbl2 = `σ₂·u₂ (${this.S[1].toFixed(2)})`;
    }

    this.canvas.drawVector(0, 0, tv1_x, tv1_y, pal.primary, lbl1, 3.5);
    this.canvas.drawVector(0, 0, tv2_x, tv2_y, pal.secondary, lbl2, 3.5);

    // Standard basis e1, e2 transformed
    const te1_x = M[0][0], te1_y = M[1][0];
    const te2_x = M[0][1], te2_y = M[1][1];
    this.canvas.drawVector(0, 0, te1_x, te1_y, isDark ? '#64748b' : '#94a3b8', 'M·e₁', 1.8);
    this.canvas.drawVector(0, 0, te2_x, te2_y, isDark ? '#64748b' : '#94a3b8', 'M·e₂', 1.8);

    // Stage label
    ctx.save();
    ctx.font = '700 13px var(--font-heading)';
    ctx.fillStyle = isDark ? '#f4f4f5' : '#000000';
    const stageDesc = s < 0.9 ? 'Этап 1: Вращение Vᵀ (базис сингулярных осей)' :
                      s < 1.9 ? 'Этап 2: Растяжение Σ (деформация по осям)' :
                      s < 2.9 ? 'Этап 3: Вращение U (финальная ориентация эллипса)' :
                                'Итог: A = U Σ Vᵀ (полная трансформация)';
    ctx.fillText(`▶ ${stageDesc}`, 20, 30);
    ctx.restore();
  }

  renderLoRA() {
    const ctx = this.canvas.ctx;
    const pal = this.getPalette();
    const isDark = this.canvas && this.canvas.mode === 'dark';
    const w = this.canvas.displayWidth || this.canvas.width || 700;
    const h = this.canvas.displayHeight || this.canvas.height || 520;

    ctx.save();
    ctx.fillStyle = isDark ? '#18181b' : '#ffffff';
    ctx.fillRect(0, 0, w, h);

    ctx.font = '800 14px var(--font-heading)';
    ctx.fillStyle = isDark ? '#f4f4f5' : '#09090b';
    ctx.fillText('Спектр сингулярных чисел σᵢ и сохранение энергии матрицы', 25, 35);

    // Top status badges overlay on canvas
    if (this.loraMetrics) {
      const badgeText = `LoRA k = ${this.rank}/6   |   Энергия: ${this.loraMetrics.energyPct}%   |   Сжатие: ${this.loraMetrics.compRatio}x`;
      ctx.font = '700 12px var(--font-mono)';
      ctx.fillStyle = isDark ? '#38bdf8' : '#0284c7';
      ctx.textAlign = 'right';
      ctx.fillText(badgeText, w - 25, 35);
      ctx.textAlign = 'left';
    }

    const sigmas = this.loraSingulars;
    const totalSq = sigmas.reduce((acc, v) => acc + v * v, 0);
    const maxVal = Math.max(...sigmas, 1.0);

    const startX = 60;
    const startY = h - 60;
    const chartW = w - 120;
    const chartH = h - 130;
    const barWidth = Math.min(60, (chartW / sigmas.length) - 20);

    ctx.strokeStyle = isDark ? '#3f3f46' : '#cbd5e1';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(startX - 10, startY);
    ctx.lineTo(startX + chartW + 10, startY);
    ctx.moveTo(startX, startY);
    ctx.lineTo(startX, startY - chartH - 20);
    ctx.stroke();

    const cumPoints = [];
    let cumEnergy = 0;

    for (let i = 0; i < sigmas.length; i++) {
      const s = sigmas[i];
      cumEnergy += s * s;
      const energyPct = totalSq > 1e-9 ? (cumEnergy / totalSq) * 100 : 100;

      const x = startX + i * (chartW / sigmas.length) + 15;
      const barH = (s / maxVal) * chartH;
      const y = startY - barH;

      const isKept = i < this.rank;

      ctx.fillStyle = isKept ? (isDark ? '#38bdf8' : '#0284c7') : (isDark ? '#3f3f46' : '#cbd5e1');
      ctx.fillRect(x, y, barWidth, barH);
      ctx.strokeStyle = isDark ? '#f4f4f5' : '#000000';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, barWidth, barH);

      ctx.fillStyle = isDark ? '#a1a1aa' : '#64748b';
      ctx.font = '700 12px var(--font-heading)';
      ctx.textAlign = 'center';
      ctx.fillText(`σ${i + 1}`, x + barWidth / 2, startY + 20);
      ctx.fillText(s.toFixed(2), x + barWidth / 2, y - 8);

      cumPoints.push({ x: x + barWidth / 2, y: startY - (energyPct / 100) * chartH, pct: energyPct });
    }

    // 90% line
    const y90 = startY - 0.9 * chartH;
    ctx.strokeStyle = '#ea580c';
    ctx.setLineDash([5, 5]);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(startX, y90);
    ctx.lineTo(startX + chartW, y90);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#ea580c';
    ctx.font = '700 11px var(--font-mono)';
    ctx.textAlign = 'right';
    ctx.fillText('Порог 90% энергии', startX + chartW, y90 - 6);

    // Cumulative curve
    if (cumPoints.length > 0) {
      ctx.strokeStyle = pal.secondary;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(cumPoints[0].x, cumPoints[0].y);
      for (let i = 1; i < cumPoints.length; i++) {
        ctx.lineTo(cumPoints[i].x, cumPoints[i].y);
      }
      ctx.stroke();

      for (const pt of cumPoints) {
        ctx.fillStyle = pal.secondary;
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = isDark ? '#ffffff' : '#000000';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    ctx.restore();
    this.updateGrids();
  }

  updateGrids() {
    const origGrid = document.getElementById('svd-orig-matrix-grid');
    const approxGrid = document.getElementById('svd-approx-matrix-grid');
    const metricsEl = document.getElementById('svd-lora-metrics');
    if (!origGrid || !approxGrid || !this.origMatrix || !this.approxMatrix) return;

    origGrid.innerHTML = '';
    approxGrid.innerHTML = '';

    const isDark = document.body.getAttribute('data-mode') === 'dark';
    const m = this.origMatrix.length;
    const n = this.origMatrix[0].length;

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        const oVal = Math.max(0, Math.min(1, this.origMatrix[i][j]));
        const aVal = Math.max(0, Math.min(1, this.approxMatrix[i][j]));

        const oCell = document.createElement('div');
        oCell.className = 'pixel-matrix-cell';
        oCell.style.backgroundColor = isDark ? `rgba(56, 189, 248, ${Math.max(0.12, oVal)})` : `rgba(2, 132, 199, ${Math.max(0.08, oVal)})`;
        oCell.style.color = oVal > 0.45 ? '#ffffff' : (isDark ? '#e4e4e7' : '#09090b');
        oCell.textContent = oVal.toFixed(1);
        origGrid.appendChild(oCell);

        const aCell = document.createElement('div');
        aCell.className = 'pixel-matrix-cell';
        aCell.style.backgroundColor = isDark ? `rgba(74, 222, 128, ${Math.max(0.12, aVal)})` : `rgba(22, 163, 74, ${Math.max(0.08, aVal)})`;
        aCell.style.color = aVal > 0.45 ? '#ffffff' : (isDark ? '#e4e4e7' : '#09090b');
        aCell.textContent = aVal.toFixed(1);
        approxGrid.appendChild(aCell);
      }
    }

    if (metricsEl && this.loraMetrics) {
      metricsEl.innerHTML = `
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
          <span class="stat-badge badge-green">Энергия: ${this.loraMetrics.energyPct}%</span>
          <span class="stat-badge badge-amber">Ошибка Фробениуса: ${this.loraMetrics.frobErr}</span>
          <span class="stat-badge badge-sky">Параметры: ${this.loraMetrics.loraParams} / ${this.loraMetrics.origParams} (Сжатие ${this.loraMetrics.compRatio}x)</span>
        </div>
      `;
    }
  }

  updateMathCard() {
    const s1 = this.S[0], s2 = this.S[1];
    const u11 = this.U[0][0], u12 = this.U[0][1];
    const u21 = this.U[1][0], u22 = this.U[1][1];
    const v11 = this.Vt[0][0], v12 = this.Vt[0][1];
    const v21 = this.Vt[1][0], v22 = this.Vt[1][1];

    const katexEl = document.getElementById('svd-latex-eq');
    if (katexEl) {
      const latex = `A = U \\Sigma V^T = \\begin{pmatrix} ${u11.toFixed(2)} & ${u12.toFixed(2)} \\\\ ${u21.toFixed(2)} & ${u22.toFixed(2)} \\end{pmatrix} \\begin{pmatrix} ${s1.toFixed(2)} & 0 \\\\ 0 & ${s2.toFixed(2)} \\end{pmatrix} \\begin{pmatrix} ${v11.toFixed(2)} & ${v12.toFixed(2)} \\\\ ${v21.toFixed(2)} & ${v22.toFixed(2)} \\end{pmatrix}`;
      if (window.katex) {
        window.katex.render(latex, katexEl, { throwOnError: false });
      } else {
        katexEl.textContent = `A = U Sigma V^T (sigma_1 = ${s1.toFixed(2)}, sigma_2 = ${s2.toFixed(2)})`;
      }
    }

    const cond = s2 > 1e-7 ? (s1 / s2).toFixed(2) : '∞ (Вырождена)';
    const frob = Math.hypot(s1, s2).toFixed(2);
    const statsEl = document.getElementById('svd-stats-info');
    if (statsEl) {
      statsEl.innerHTML = `
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.4rem;">
          <span class="stat-badge badge-sky">σ₁ = ${s1.toFixed(2)}, σ₂ = ${s2.toFixed(2)}</span>
          <span class="stat-badge badge-amber">Число обусловленности κ = ${cond}</span>
          <span class="stat-badge badge-green">||A||_F = ${frob}</span>
        </div>
        <p style="font-size:0.82rem; color:var(--text-muted); line-height:1.5;">
          <b>Теорема Эккарта-Янга:</b> усеченное разложение <i>Aₖ = Σ σᵢ uᵢ vᵢᵀ</i> дает математически наилучшую возможную низкоранговую аппроксимацию матрицы по норме Фробениуса. На этом принципе построена технология <b>LoRA (Low-Rank Adaptation)</b> для эффективного дообучения больших языковых моделей (LLM).
        </p>
      `;
    }
  }
}

// ============================================================
// 8. MODULE: VECTOR EMBEDDINGS, COSINE SIMILARITY & RAG
// ============================================================
class ModuleEmbeddings {
  constructor(canvas) {
    this.canvas = canvas;
    this.mode = 'rag'; // 'rag' | 'analogy'
    this.topK = 3;
    this.queryAngle = 45.0; // degrees
    this.queryLen = 2.5;
    this.paletteId = 1;
    this.modeTheme = 'light';
    this.currentAnalogy = 'king';

    this.concepts = [
      { label: "Transformer", category: "LLM & Архитектуры", angle: 32, r: 2.8 },
      { label: "Attention (Внимание)", category: "LLM & Архитектуры", angle: 45, r: 2.5 },
      { label: "RAG & Поиск", category: "LLM & Архитектуры", angle: 58, r: 2.7 },
      { label: "Векторные БД", category: "Инженерия данных", angle: 72, r: 2.4 },
      { label: "Косинусное сходство", category: "Метрики сходства", angle: 88, r: 2.6 },
      { label: "Метод PCA", category: "Редукция размерности", angle: 140, r: 2.5 },
      { label: "SVD Разложение", category: "Редукция размерности", angle: 155, r: 2.7 },
      { label: "LoRA Адаптация", category: "Редукция размерности", angle: 172, r: 2.3 },
      { label: "МНК Регрессия", category: "Оптимизация", angle: 220, r: 2.6 },
      { label: "Градиентный спуск", category: "Оптимизация", angle: 245, r: 2.5 },
      { label: "Функция потерь MSE", category: "Оптимизация", angle: 260, r: 2.4 },
      { label: "Умножение матриц", category: "Линейная алгебра", angle: 310, r: 2.5 },
      { label: "Собственные векторы", category: "Линейная алгебра", angle: 330, r: 2.7 }
    ];

    this.analogies = {
      king: {
        title: "Король - Мужчина + Женщина = Королева",
        a: { label: "Король", vec: [2.0, 1.2] },
        b: { label: "Мужчина", vec: [1.8, -0.6] },
        c: { label: "Женщина", vec: [0.3, 1.4] },
        target: { label: "Королева", vec: [0.5, 3.2] },
        desc: "Классический пример Mikolov et al. (2013). Направление Мужчина -> Женщина кодирует семантическую ось биологического пола."
      },
      deeplearning: {
        title: "Нейросеть - Линейный + Глубокий = Transformer",
        a: { label: "Нейросеть", vec: [1.2, 1.8] },
        b: { label: "Линейный", vec: [-1.2, 0.8] },
        c: { label: "Глубокий", vec: [0.6, 2.2] },
        target: { label: "Transformer", vec: [3.0, 3.2] },
        desc: "Векторные модификаторы позволяют добавлять концептуальные свойства моделям машинного обучения."
      },
      tensor: {
        title: "Матрица - 2D + 3D = Тензор",
        a: { label: "Матрица", vec: [2.2, 0.6] },
        b: { label: "2D", vec: [1.6, -1.0] },
        c: { label: "3D", vec: [-0.4, 1.6] },
        target: { label: "Тензор", vec: [0.2, 3.2] },
        desc: "Ось размерности данных транслирует 2D таблицы в многомерные тензоры PyTorch / TensorFlow."
      }
    };

    this.customVectors = {
      king: { a: [2.0, 1.2], b: [1.8, -0.6], c: [0.3, 1.4] },
      deeplearning: { a: [1.2, 1.8], b: [-1.2, 0.8], c: [0.6, 2.2] },
      tensor: { a: [2.2, 0.6], b: [1.6, -1.0], c: [-0.4, 1.6] }
    };

    // Animation state
    this.isAnimating = false;
    this.animProgress = 1.0;
    this.animFrameId = null;
    this.animStartTime = null;
    this.animDuration = 2800; // ms

    this.initCanvasHandles();
  }

  setMode(mode) {
    this.mode = mode;
    this.stopAnimation();
    this.updateHandles();
    this.render();
    if (mode === 'rag') {
      this.updateRAGTable();
    } else {
      this.updateAnalogyUI();
    }
  }

  setAnalogy(key) {
    if (!this.analogies[key]) return;
    this.stopAnimation();
    this.currentAnalogy = key;
    this.animProgress = 1.0;
    this.updateHandles();
    this.render();
    this.updateAnalogyUI();
  }

  getCurrentAnalogyVectors() {
    const preset = this.analogies[this.currentAnalogy];
    const custom = (this.customVectors && this.customVectors[this.currentAnalogy]) ? this.customVectors[this.currentAnalogy] : {
      a: [...preset.a.vec],
      b: [...preset.b.vec],
      c: [...preset.c.vec]
    };
    return {
      a: custom.a,
      b: custom.b,
      c: custom.c,
      target: preset.target.vec,
      title: preset.title,
      labels: {
        a: preset.a.label,
        b: preset.b.label,
        c: preset.c.label,
        target: preset.target.label
      },
      desc: preset.desc
    };
  }

  resetAnalogyVectors() {
    this.stopAnimation();
    const preset = this.analogies[this.currentAnalogy];
    if (preset) {
      this.customVectors[this.currentAnalogy] = {
        a: [...preset.a.vec],
        b: [...preset.b.vec],
        c: [...preset.c.vec]
      };
      this.animProgress = 1.0;
      this.updateHandles();
      this.render();
      this.updateAnalogyUI();
    }
  }

  togglePlayAnalogy() {
    if (this.isAnimating) {
      this.stopAnimation();
    } else {
      this.startAnimation();
    }
  }

  startAnimation() {
    this.isAnimating = true;
    this.animProgress = 0.0;
    this.animStartTime = null;
    const playBtn = document.getElementById('btn-emb-analogy-play');
    if (playBtn) playBtn.innerHTML = '⏸ Пауза';
    this.canvas.clearHandles();

    const step = (timestamp) => {
      if (!this.isAnimating) return;
      if (!this.animStartTime) this.animStartTime = timestamp;
      const elapsed = timestamp - this.animStartTime;
      this.animProgress = Math.min(1.0, elapsed / this.animDuration);
      this.render();

      if (this.animProgress < 1.0) {
        this.animFrameId = requestAnimationFrame(step);
      } else {
        this.isAnimating = false;
        if (playBtn) playBtn.innerHTML = '▶ Анимировать сборку';
        this.updateHandles();
        this.render();
      }
    };
    this.animFrameId = requestAnimationFrame(step);
  }

  stopAnimation() {
    this.isAnimating = false;
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    const playBtn = document.getElementById('btn-emb-analogy-play');
    if (playBtn) playBtn.innerHTML = '▶ Анимировать сборку';
    this.updateHandles();
  }

  setPalette(pId, mode = 'light') {
    this.paletteId = parseInt(pId, 10) || 1;
    this.modeTheme = mode;
    this.render();
    if (this.mode === 'rag') {
      this.updateRAGTable();
    }
  }

  getPalette() {
    return window.getAppPalette(this.paletteId, (this.canvas ? this.canvas.mode : this.modeTheme) || 'light');
  }

  initCanvasHandles() {
    this.updateHandles();
    this.canvas.onHandleDrag = (id, x, y) => {
      if (id === 'emb_query') {
        let deg = Math.round((Math.atan2(y, x) * 180) / Math.PI);
        if (deg < 0) deg += 360;
        this.queryAngle = deg;
        this.queryLen = Math.max(1.0, Math.min(4.0, Math.hypot(x, y)));
        const slider = document.getElementById('emb-query-angle-slider');
        const valEl = document.getElementById('emb-query-angle-val');
        if (slider) slider.value = deg;
        if (valEl) valEl.textContent = `${deg}°`;
        this.render();
        this.updateRAGTable();
      } else if (id === 'emb_vec_a') {
        if (!this.customVectors[this.currentAnalogy]) this.customVectors[this.currentAnalogy] = {};
        this.customVectors[this.currentAnalogy].a = [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
        this.render();
        this.updateAnalogyUI();
      } else if (id === 'emb_vec_b') {
        if (!this.customVectors[this.currentAnalogy]) this.customVectors[this.currentAnalogy] = {};
        this.customVectors[this.currentAnalogy].b = [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
        this.render();
        this.updateAnalogyUI();
      } else if (id === 'emb_vec_c') {
        if (!this.customVectors[this.currentAnalogy]) this.customVectors[this.currentAnalogy] = {};
        this.customVectors[this.currentAnalogy].c = [Math.round(x * 10) / 10, Math.round(y * 10) / 10];
        this.render();
        this.updateAnalogyUI();
      }
    };
    this.canvas.onPan = () => this.render();
    this.canvas.onZoom = () => this.render();
  }

  updateHandles() {
    this.canvas.clearHandles();
    if (this.mode === 'rag') {
      const pal = this.getPalette();
      const rad = (this.queryAngle * Math.PI) / 180;
      const qx = Math.cos(rad) * this.queryLen;
      const qy = Math.sin(rad) * this.queryLen;
      this.canvas.addHandle('emb_query', qx, qy, 9, pal.secondary, 'q');
    } else if (this.mode === 'analogy' && !this.isAnimating) {
      const vecs = this.getCurrentAnalogyVectors();
      this.canvas.addHandle('emb_vec_a', vecs.a[0], vecs.a[1], 8, '#0284c7', 'A');
      this.canvas.addHandle('emb_vec_b', vecs.b[0], vecs.b[1], 8, '#ea580c', 'B');
      this.canvas.addHandle('emb_vec_c', vecs.c[0], vecs.c[1], 8, '#16a34a', 'C');
    }
  }

  updateQueryHandle() {
    this.updateHandles();
  }

  getRankedConcepts() {
    const qRad = (this.queryAngle * Math.PI) / 180;
    const qx = Math.cos(qRad) * this.queryLen;
    const qy = Math.sin(qRad) * this.queryLen;
    const qNorm = Math.hypot(qx, qy);

    const scored = this.concepts.map(c => {
      const cRad = (c.angle * Math.PI) / 180;
      const cx = Math.cos(cRad) * c.r;
      const cy = Math.sin(cRad) * c.r;
      const cNorm = Math.hypot(cx, cy);

      const dot = qx * cx + qy * cy;
      const cosSim = (qNorm > 1e-7 && cNorm > 1e-7) ? dot / (qNorm * cNorm) : 0;
      const diffDeg = Math.abs(this.queryAngle - c.angle) % 360;
      const angleErr = Math.min(diffDeg, 360 - diffDeg);

      return {
        ...c,
        cx,
        cy,
        cosSim,
        angleErr
      };
    });

    scored.sort((a, b) => b.cosSim - a.cosSim);
    return scored;
  }

  addCustomConcept(label) {
    if (!label || !label.trim()) return;
    const newConcept = {
      label: label.trim(),
      category: "Пользовательский",
      angle: this.queryAngle,
      r: this.queryLen
    };
    this.concepts.push(newConcept);
    this.render();
    this.updateRAGTable();
  }

  render() {
    this.canvas.clear();

    if (this.mode === 'rag') {
      this.renderRAG();
    } else {
      this.renderAnalogy();
    }
  }

  renderRAG() {
    this.canvas.drawGrid();
    const pal = this.getPalette();
    const isDark = this.canvas && this.canvas.mode === 'dark';
    const ctx = this.canvas.ctx;
    const haloColor = isDark ? '#18181b' : '#ffffff';

    // Unit circle (Semantic Normalized Manifold)
    this.canvas.drawCircle(0, 0, 2.5, isDark ? 'rgba(56, 189, 248, 0.06)' : 'rgba(2, 132, 199, 0.05)', isDark ? '#334155' : '#cbd5e1', 1.5, true);

    const ranked = this.getRankedConcepts();
    const topKSet = new Set(ranked.slice(0, this.topK).map(c => c.label));

    // Query vector q
    const qRad = (this.queryAngle * Math.PI) / 180;
    const qx = Math.cos(qRad) * this.queryLen;
    const qy = Math.sin(qRad) * this.queryLen;
    this.canvas.drawVector(0, 0, qx, qy, pal.secondary, `Query q (${this.queryAngle}°)`, 3.5);

    // Draw connection lines to Top-K
    for (let i = 0; i < Math.min(this.topK, ranked.length); i++) {
      const item = ranked[i];
      const p1 = this.canvas.mathToCanvas(qx, qy);
      const p2 = this.canvas.mathToCanvas(item.cx, item.cy);

      ctx.save();
      ctx.strokeStyle = pal.primary;
      ctx.lineWidth = 1.8;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(p1.px, p1.py);
      ctx.lineTo(p2.px, p2.py);
      ctx.stroke();
      ctx.restore();
    }

    // Draw all concept points with halo and clean labels
    for (let i = 0; i < ranked.length; i++) {
      const item = ranked[i];
      const isTopK = topKSet.has(item.label);
      const { px, py } = this.canvas.mathToCanvas(item.cx, item.cy);

      ctx.save();
      ctx.beginPath();
      ctx.arc(px, py, isTopK ? 7 : 4.5, 0, 2 * Math.PI);
      ctx.fillStyle = isTopK ? pal.primary : (isDark ? '#475569' : '#94a3b8');
      ctx.fill();
      ctx.strokeStyle = isDark ? '#ffffff' : '#000000';
      ctx.lineWidth = isTopK ? 2 : 1;
      ctx.stroke();

      ctx.font = isTopK ? 'bold 12px var(--font-heading)' : '600 11px var(--font-heading)';
      ctx.fillStyle = isTopK ? (isDark ? '#38bdf8' : '#0284c7') : (isDark ? '#94a3b8' : '#64748b');
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      const labelText = isTopK ? `#${i + 1} ${item.label} (${item.cosSim.toFixed(2)})` : item.label;
      ctx.strokeStyle = haloColor;
      ctx.lineWidth = 3.5;
      ctx.strokeText(labelText, px + 10, py);
      ctx.fillText(labelText, px + 10, py);
      ctx.restore();
    }

    // Draw arc between query and Top-1 with angle text
    if (ranked.length > 0) {
      const best = ranked[0];
      const bestRad = (best.angle * Math.PI) / 180;
      this.canvas.drawAngleArc(0, 0, 1.2, bestRad, qRad, pal.secondary, 2.5);

      // Label along the arc
      let diff = (qRad - bestRad) % (2 * Math.PI);
      if (diff > Math.PI) diff -= 2 * Math.PI;
      if (diff < -Math.PI) diff += 2 * Math.PI;
      const midRad = bestRad + diff / 2;
      const { px: arcPx, py: arcPy } = this.canvas.mathToCanvas(1.45 * Math.cos(midRad), 1.45 * Math.sin(midRad));

      ctx.save();
      ctx.font = 'bold 11px var(--font-mono)';
      ctx.fillStyle = pal.secondary;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.strokeStyle = haloColor;
      ctx.lineWidth = 3.5;
      const arcText = `θ = ${best.angleErr.toFixed(1)}°`;
      ctx.strokeText(arcText, arcPx, arcPy);
      ctx.fillText(arcText, arcPx, arcPy);
      ctx.restore();

      // Top title indicator
      ctx.save();
      ctx.font = '700 13px var(--font-heading)';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.strokeStyle = haloColor;
      ctx.lineWidth = 3.5;
      ctx.fillStyle = isDark ? '#38bdf8' : '#0284c7';
      const titleText = `▶ Поиск RAG: Top-1 = ${best.label} (cos θ = ${best.cosSim.toFixed(4)}, Δθ = ${best.angleErr.toFixed(1)}°)`;
      ctx.strokeText(titleText, 18, 16);
      ctx.fillText(titleText, 18, 16);
      ctx.restore();

      // Footer hint
      const hintEl = document.getElementById('emb-footer-hint');
      if (hintEl) {
        hintEl.innerHTML = `💡 Top-1 результат: <b>${best.label}</b> (cos θ = <b>${best.cosSim.toFixed(4)}</b>, угол Δθ = ${best.angleErr.toFixed(1)}°). Перетаскивайте маркер <b>q</b> мышкой прямо по плоскости!`;
      }
    }

    this.updateQueryHandle();
    this.canvas.drawHandles();
  }

  renderAnalogy() {
    this.canvas.drawGrid();
    const pal = this.getPalette();
    const isDark = this.canvas && this.canvas.mode === 'dark';
    const ctx = this.canvas.ctx;
    const haloColor = isDark ? '#18181b' : '#ffffff';

    const vecs = this.getCurrentAnalogyVectors();
    const A = vecs.a;
    const B = vecs.b;
    const C = vecs.c;
    const pred = [A[0] - B[0] + C[0], A[1] - B[1] + C[1]];
    const target = vecs.target;

    const t = this.animProgress; // 0.0 to 1.0

    // 1. Base reference points & faint vectors for B and C from origin
    if (t >= 0.05) {
      this.canvas.drawPolygon([[0, 0], [B[0], B[1]]], null, isDark ? '#52525b' : '#cbd5e1', 1.5, true);
      const pB = this.canvas.mathToCanvas(B[0], B[1]);
      ctx.save();
      ctx.beginPath();
      ctx.arc(pB.px, pB.py, 4, 0, 2 * Math.PI);
      ctx.fillStyle = '#ea580c';
      ctx.fill();
      ctx.font = '600 11px var(--font-heading)';
      ctx.fillStyle = isDark ? '#fb923c' : '#c2410c';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.strokeStyle = haloColor;
      ctx.lineWidth = 3;
      ctx.strokeText(` ${vecs.labels.b} (B)`, pB.px + 8, pB.py);
      ctx.fillText(` ${vecs.labels.b} (B)`, pB.px + 8, pB.py);
      ctx.restore();

      this.canvas.drawPolygon([[0, 0], [C[0], C[1]]], null, isDark ? '#52525b' : '#cbd5e1', 1.5, true);
      const pC = this.canvas.mathToCanvas(C[0], C[1]);
      ctx.save();
      ctx.beginPath();
      ctx.arc(pC.px, pC.py, 4, 0, 2 * Math.PI);
      ctx.fillStyle = '#16a34a';
      ctx.fill();
      ctx.font = '600 11px var(--font-heading)';
      ctx.fillStyle = isDark ? '#4ade80' : '#15803d';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.strokeStyle = haloColor;
      ctx.lineWidth = 3;
      ctx.strokeText(` ${vecs.labels.c} (C)`, pC.px + 8, pC.py);
      ctx.fillText(` ${vecs.labels.c} (C)`, pC.px + 8, pC.py);
      ctx.restore();
    }

    // 2. Vector A: (0, 0) -> A (animated in t in [0.0, 0.25])
    const p1 = Math.min(1.0, Math.max(0.0, t / 0.25));
    if (p1 > 0.02) {
      const curAx = A[0] * p1;
      const curAy = A[1] * p1;
      this.canvas.drawVector(0, 0, curAx, curAy, '#0284c7', p1 >= 0.9 ? `${vecs.labels.a} (A)` : '', 3.5);
    }

    // 3. Vector -B: from A to A - B (animated in t in [0.25, 0.50])
    const amb = [A[0] - B[0], A[1] - B[1]];
    if (t > 0.25) {
      const p2 = Math.min(1.0, Math.max(0.0, (t - 0.25) / 0.25));
      const curBx = A[0] - B[0] * p2;
      const curBy = A[1] - B[1] * p2;
      this.canvas.drawVector(A[0], A[1], curBx, curBy, '#ea580c', p2 >= 0.9 ? `− ${vecs.labels.b} (−B)` : '', 2.8);
    }

    // 4. Vector +C: from A - B to A - B + C (animated in t in [0.50, 0.75])
    if (t > 0.50) {
      const p3 = Math.min(1.0, Math.max(0.0, (t - 0.50) / 0.25));
      const curCx = amb[0] + C[0] * p3;
      const curCy = amb[1] + C[1] * p3;
      this.canvas.drawVector(amb[0], amb[1], curCx, curCy, '#16a34a', p3 >= 0.9 ? `+ ${vecs.labels.c} (+C)` : '', 2.8);
    }

    // 5. Result vector: (0, 0) -> pred (animated in t in [0.75, 1.00])
    if (t > 0.75) {
      const p4 = Math.min(1.0, Math.max(0.0, (t - 0.75) / 0.25));
      const curRx = pred[0] * p4;
      const curRy = pred[1] * p4;
      this.canvas.drawVector(0, 0, curRx, curRy, '#f59e0b', p4 >= 0.9 ? `A − B + C` : '', 3.8);
    }

    // 6. Target concept point (with glowing halo)
    const { px, py } = this.canvas.mathToCanvas(target[0], target[1]);
    ctx.save();
    // Halo ring
    ctx.beginPath();
    ctx.arc(px, py, t >= 0.95 ? 16 : 12, 0, 2 * Math.PI);
    ctx.fillStyle = isDark ? 'rgba(52, 211, 153, 0.25)' : 'rgba(16, 185, 129, 0.22)';
    ctx.fill();

    // Target circle
    ctx.beginPath();
    ctx.arc(px, py, 8, 0, 2 * Math.PI);
    ctx.fillStyle = '#10b981';
    ctx.fill();
    ctx.strokeStyle = isDark ? '#ffffff' : '#000000';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Target label (offset nicely so it never overlaps arrow)
    ctx.font = '800 13px var(--font-heading)';
    ctx.fillStyle = isDark ? '#34d399' : '#059669';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle = haloColor;
    ctx.lineWidth = 3.5;
    const tgtText = `★ Цель: ${vecs.labels.target} [${target[0].toFixed(1)}, ${target[1].toFixed(1)}]`;
    ctx.strokeText(tgtText, px + 14, py - 12);
    ctx.fillText(tgtText, px + 14, py - 12);
    ctx.restore();

    // 7. Top indicator
    ctx.save();
    ctx.font = '700 13px var(--font-heading)';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.strokeStyle = haloColor;
    ctx.lineWidth = 3.5;
    ctx.fillStyle = isDark ? '#34d399' : '#059669';
    const topTitle = `▶ Арифметика Word2Vec: ${vecs.title}`;
    ctx.strokeText(topTitle, 18, 16);
    ctx.fillText(topTitle, 18, 16);
    ctx.restore();

    // 8. Footer hint with similarity check
    const predNorm = Math.hypot(pred[0], pred[1]);
    const tgtNorm = Math.hypot(target[0], target[1]);
    const dot = pred[0] * target[0] + pred[1] * target[1];
    const cosWithTarget = (predNorm > 1e-6 && tgtNorm > 1e-6) ? (dot / (predNorm * tgtNorm)) : 0;

    const hintEl = document.getElementById('emb-footer-hint');
    if (hintEl) {
      if (cosWithTarget >= 0.98) {
        hintEl.innerHTML = `💡 <b>${vecs.title}</b>: сумма <b>A − B + C = [${pred[0].toFixed(2)}, ${pred[1].toFixed(2)}]</b> совпадает с <b>${vecs.labels.target}</b>! (cos = <b>${cosWithTarget.toFixed(4)}</b>)`;
      } else {
        hintEl.innerHTML = `💡 Вектор <b>A − B + C = [${pred[0].toFixed(2)}, ${pred[1].toFixed(2)}]</b>. Сходство с <b>${vecs.labels.target}</b>: <b>${cosWithTarget.toFixed(4)}</b>. Двигайте маркеры A, B, C!`;
      }
    }

    // Draw handles if not animating
    if (!this.isAnimating) {
      this.updateHandles();
      this.canvas.drawHandles();
    }
  }

  updateRAGTable() {
    const table = document.getElementById('emb-rag-table');
    if (!table) return;
    const tbody = table.querySelector('tbody');
    if (!tbody) return;

    const ranked = this.getRankedConcepts();
    const isDark = (this.canvas && this.canvas.mode === 'dark') || this.modeTheme === 'dark';

    const getCategoryBadge = (cat) => {
      let bg = '#e0f2fe';
      let fg = '#0369a1';
      if (cat.includes('LLM') || cat.includes('Архитектуры')) {
        bg = isDark ? 'rgba(56, 189, 248, 0.2)' : '#e0f2fe';
        fg = isDark ? '#38bdf8' : '#0284c7';
      } else if (cat.includes('Данных') || cat.includes('БД')) {
        bg = isDark ? 'rgba(168, 85, 247, 0.2)' : '#f3e8ff';
        fg = isDark ? '#c084fc' : '#7e22ce';
      } else if (cat.includes('Метрики') || cat.includes('сходства')) {
        bg = isDark ? 'rgba(34, 197, 94, 0.2)' : '#dcfce7';
        fg = isDark ? '#4ade80' : '#15803d';
      } else if (cat.includes('Редукция')) {
        bg = isDark ? 'rgba(236, 72, 153, 0.2)' : '#fce7f3';
        fg = isDark ? '#f472b6' : '#be185d';
      } else if (cat.includes('Оптимизация')) {
        bg = isDark ? 'rgba(245, 158, 11, 0.2)' : '#fef3c7';
        fg = isDark ? '#fbbf24' : '#b45309';
      } else if (cat.includes('Линейная')) {
        bg = isDark ? 'rgba(99, 102, 241, 0.2)' : '#e0e7ff';
        fg = isDark ? '#818cf8' : '#4338ca';
      } else {
        bg = isDark ? 'rgba(20, 184, 166, 0.2)' : '#ccfbf1';
        fg = isDark ? '#2dd4bf' : '#0f766e';
      }
      return `<span style="display:inline-block; font-size:0.7rem; font-weight:700; padding:2px 6px; border-radius:4px; background:${bg}; color:${fg}; white-space:nowrap;">${cat}</span>`;
    };

    let html = '';
    ranked.forEach((item, index) => {
      const rank = index + 1;
      const isTopK = rank <= this.topK;

      let cosColor = isDark ? '#f87171' : '#dc2626';
      if (item.cosSim >= 0.85) {
        cosColor = isDark ? '#4ade80' : '#15803d';
      } else if (item.cosSim >= 0.5) {
        cosColor = isDark ? '#38bdf8' : '#0284c7';
      } else if (item.cosSim >= 0.0) {
        cosColor = isDark ? '#fbbf24' : '#d97706';
      }

      const rowClass = isTopK ? 'top-k-row' : '';
      const star = isTopK ? '★ ' : '';

      html += `
        <tr class="${rowClass}" data-angle="${item.angle}" style="cursor:pointer;" title="Нажмите, чтобы направить вектор запроса q к '${item.label}' (${item.angle}°)">
          <td style="text-align:center; font-family:var(--font-mono); font-weight:${isTopK ? '800' : '600'}; color:${isTopK ? (isDark ? '#38bdf8' : '#0284c7') : 'inherit'};">
            ${star}${rank}
          </td>
          <td style="font-weight:${isTopK ? '800' : '500'};">
            ${item.label}
          </td>
          <td>
            ${getCategoryBadge(item.category)}
          </td>
          <td style="text-align:right; font-family:var(--font-mono); font-weight:750; color:${cosColor};">
            ${item.cosSim.toFixed(4)}
          </td>
          <td style="text-align:right; font-family:var(--font-mono); font-size:0.78rem; color:var(--text-muted);">
            ${item.angleErr.toFixed(1)}°
          </td>
        </tr>
      `;
    });

    tbody.innerHTML = html;

    // Attach click listeners to rows to quickly align query vector with a concept
    tbody.querySelectorAll('tr').forEach(tr => {
      tr.addEventListener('click', () => {
        const angle = parseFloat(tr.dataset.angle);
        if (!isNaN(angle)) {
          this.queryAngle = angle;
          const slider = document.getElementById('emb-query-angle-slider');
          const valEl = document.getElementById('emb-query-angle-val');
          if (slider) slider.value = angle;
          if (valEl) valEl.textContent = `${angle}°`;
          this.render();
          this.updateRAGTable();
        }
      });
    });
  }

  updateAnalogyUI() {
    const vecs = this.getCurrentAnalogyVectors();
    const eqBox = document.getElementById('emb-equation-box');
    if (eqBox) {
      eqBox.innerHTML = `
        <span class="analogy-term term-pos">${vecs.labels.a}</span>
        <span class="analogy-op">−</span>
        <span class="analogy-term term-neg">${vecs.labels.b}</span>
        <span class="analogy-op">+</span>
        <span class="analogy-term term-pos">${vecs.labels.c}</span>
        <span class="analogy-op">≈</span>
        <span class="analogy-term term-res">${vecs.labels.target}</span>
      `;
    }

    const card = document.getElementById('emb-analogy-result-card');
    if (card) {
      const pred = [vecs.a[0] - vecs.b[0] + vecs.c[0], vecs.a[1] - vecs.b[1] + vecs.c[1]];
      const predNorm = Math.hypot(pred[0], pred[1]);
      const tgtNorm = Math.hypot(vecs.target[0], vecs.target[1]);
      const dot = pred[0] * vecs.target[0] + pred[1] * vecs.target[1];
      const cosWithTarget = (predNorm > 1e-6 && tgtNorm > 1e-6) ? (dot / (predNorm * tgtNorm)) : 0;
      const isClose = cosWithTarget >= 0.98;

      card.innerHTML = `
        <div style="font-weight:800; font-size:0.95rem; margin-bottom:0.4rem; color:${isClose ? '#15803d' : '#0284c7'};">
          ${isClose ? '🎯 Точное совпадение с целью:' : '📍 Текущая векторная сумма:'} [${pred[0].toFixed(2)}, ${pred[1].toFixed(2)}]
          <span style="font-size:0.8rem; font-weight:700; margin-left:0.4rem; color:var(--text-muted);">(cos = ${cosWithTarget.toFixed(4)})</span>
        </div>
        <div style="font-family:var(--font-mono); font-size:0.8rem; background:rgba(0,0,0,0.04); padding:6px 10px; border-radius:6px; margin-bottom:0.5rem; line-height:1.6;">
          <div><b style="color:#0284c7;">A (${vecs.labels.a}):</b> [${vecs.a[0].toFixed(1)}, ${vecs.a[1].toFixed(1)}]</div>
          <div><b style="color:#ea580c;">B (${vecs.labels.b}):</b> [${vecs.b[0].toFixed(1)}, ${vecs.b[1].toFixed(1)}]</div>
          <div><b style="color:#16a34a;">C (${vecs.labels.c}):</b> [${vecs.c[0].toFixed(1)}, ${vecs.c[1].toFixed(1)}]</div>
          <div style="color:${isClose ? '#15803d' : '#d97706'}; font-weight:800;"><b>A − B + C:</b> [${pred[0].toFixed(2)}, ${pred[1].toFixed(2)}] (Цель: ${vecs.labels.target})</div>
        </div>
        <p style="font-size:0.82rem; color:var(--text-muted); line-height:1.5; margin:0;">
          ${vecs.desc}
        </p>
      `;
    }
  }

  updateMathCard() {
    const katexEl = document.getElementById('emb-latex-formula');
    if (katexEl) {
      const latex = `\\text{CosineSimilarity}(u, v) = \\frac{u \\cdot v}{\\|u\\| \\|v\\|} = \\frac{\\sum_{i=1}^d u_i v_i}{\\sqrt{\\sum_{i=1}^d u_i^2} \\sqrt{\\sum_{i=1}^d v_i^2}} = \\cos(\\theta)`;
      if (window.katex) {
        window.katex.render(latex, katexEl, { throwOnError: false });
      } else {
        katexEl.textContent = "cos(theta) = (u . v) / (||u|| * ||v||)";
      }
    }

    const statsEl = document.getElementById('emb-stats-info');
    if (statsEl) {
      statsEl.innerHTML = `
        <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.4rem;">
          <span class="stat-badge badge-green">Диапазон: [-1.0, 1.0]</span>
          <span class="stat-badge badge-sky">Инвариантность к длине ||v||</span>
          <span class="stat-badge badge-amber">Векторный поиск RAG</span>
        </div>
        <p style="font-size:0.82rem; color:var(--text-muted); line-height:1.5;">
          Косинусное сходство измеряет угол между семантическими векторами текста, игнорируя их физическую длину (количество слов). Именно оно используется в <b>векторных базах данных (Chroma, Qdrant, Pinecone, FAISS)</b> для поиска наиболее релевантных фрагментов документации в архитектуре <b>RAG</b>.
        </p>
      `;
    }
  }
}
