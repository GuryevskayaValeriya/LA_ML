/**
 * LA_ML: Main Application Orchestrator & Event Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize Canvases
  const canvasVectors = new InteractiveCanvas(document.getElementById('canvas-vectors'));
  const canvas1 = new InteractiveCanvas(document.getElementById('canvas-module1'));
  const canvas2 = new InteractiveCanvas(document.getElementById('canvas-module2'));
  const canvas3 = new InteractiveCanvas(document.getElementById('canvas-module3'));
  const canvas4 = new InteractiveCanvas(document.getElementById('canvas-module4'));
  const canvasSVD = new InteractiveCanvas(document.getElementById('canvas-svd'));
  const canvasEmbeddings = new InteractiveCanvas(document.getElementById('canvas-embeddings'));
  const canvasSVM = new InteractiveCanvas(document.getElementById('canvas-svm'));
  const canvasAttention = new InteractiveCanvas(document.getElementById('canvas-attention'));
  const canvasHessian = new InteractiveCanvas(document.getElementById('canvas-hessian'));

  // 2. Initialize Interactive Modules
  const modMatMul = new ModuleMatMul();
  const modVectors = new ModuleVectorGeometry(canvasVectors);
  const modTransform = new ModuleTransform(canvas1);
  const modPCA = new ModulePCA(canvas2);
  const modLS = new ModuleLeastSquares(canvas3);
  const modTrainer = new ModuleTrainer(canvas4);
  const modSVD = new ModuleSVD(canvasSVD);
  const modEmbeddings = new ModuleEmbeddings(canvasEmbeddings);
  const modSVM = new ModuleSVM(canvasSVM);
  const modAttention = new ModuleAttention(canvasAttention);
  const modHessian = new ModuleHessian(canvasHessian);
  window.activeModuleAttention = modAttention;
  const datasetManager = new DatasetManager();
  const reportGen = new ReportGenerator({
    transform: modTransform,
    pca: modPCA,
    leastSquares: modLS,
    svm: modSVM,
    hessian: modHessian,
    attention: modAttention,
    trainer: modTrainer
  });

  let currentActiveTab = 'tab-overview';

  // 3. Unified Tab Switching Controller
  const switchTab = (tabId) => {
    currentActiveTab = tabId;
    const pills = document.querySelectorAll('.nav-pill, .tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    pills.forEach(p => {
      if (p.dataset.tab === tabId) {
        p.classList.add('active');
      } else {
        p.classList.remove('active');
      }
    });

    tabContents.forEach(c => {
      if (c.id === tabId) {
        c.classList.add('active');
      } else {
        c.classList.remove('active');
      }
    });

    // Trigger canvas resize and re-render on tab switch
    setTimeout(() => {
      if (tabId === 'tab-matmul') {
        modMatMul.render();
      } else if (tabId === 'tab-vectors') {
        canvasVectors.resize();
        modVectors.render();
        modVectors.updateMathCard();
      } else if (tabId === 'tab-module1') {
        canvas1.resize();
        modTransform.render();
        modTransform.updateMathCard();
      } else if (tabId === 'tab-svm') {
        canvasSVM.resize();
        modSVM.render();
        modSVM.updateMathCard();
      } else if (tabId === 'tab-attention') {
        canvasAttention.resize();
        modAttention.render();
        modAttention.updateMatrixUI();
      } else if (tabId === 'tab-hessian') {
        canvasHessian.resize();
        modHessian.render();
        modHessian.updateMathCard();
      } else if (tabId === 'tab-module2') {
        canvas2.resize();
        modPCA.render();
        modPCA.updateMathCard();
      } else if (tabId === 'tab-module3') {
        canvas3.resize();
        modLS.render();
        modLS.updateMathCard();
        modLS.updateStatusUI();
      } else if (tabId === 'tab-svd') {
        canvasSVD.resize();
        modSVD.render();
        modSVD.updateMathCard();
        modSVD.updateGrids();
      } else if (tabId === 'tab-embeddings') {
        canvasEmbeddings.resize();
        modEmbeddings.render();
        modEmbeddings.updateMathCard();
        modEmbeddings.updateRAGTable();
        modEmbeddings.updateAnalogyUI();
      } else if (tabId === 'tab-module4') {
        canvas4.resize();
        modTrainer.render();
      }
      renderAllFormulas();
    }, 50);

    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Nav pills listeners
  document.querySelectorAll('.nav-pill, .tab-btn').forEach(tab => {
    tab.addEventListener('click', () => {
      switchTab(tab.dataset.tab);
    });
  });

  // Hero CTA button listener
  const heroBtn = document.getElementById('btn-hero-start');
  if (heroBtn) {
    heroBtn.addEventListener('click', () => switchTab('tab-matmul'));
  }

  // Feature cards "Открыть модуль" buttons
  document.querySelectorAll('[data-go-tab]').forEach(btn => {
    btn.addEventListener('click', () => {
      switchTab(btn.dataset.goTab);
    });
  });

  const renderAllFormulas = () => {
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false }
        ],
        throwOnError: false
      });
    }
  };
  setTimeout(renderAllFormulas, 250);

  // Window Resize
  window.addEventListener('resize', () => {
    canvasVectors.resize(); modVectors.render();
    canvas1.resize(); modTransform.render();
    canvas2.resize(); modPCA.render();
    canvas3.resize(); modLS.render();
    canvas4.resize(); modTrainer.render();
    canvasSVD.resize(); modSVD.render();
    canvasEmbeddings.resize(); modEmbeddings.render();
    canvasSVM.resize(); modSVM.render();
    canvasAttention.resize(); modAttention.render();
    canvasHessian.resize(); modHessian.render();
  });

  // ==========================================
  // MODULE: MATRIX MULTIPLICATION CONTROLS
  // ==========================================
  const btnMatmulPlay = document.getElementById('btn-matmul-play');
  if (btnMatmulPlay) {
    btnMatmulPlay.addEventListener('click', () => modMatMul.togglePlay());
  }

  const btnMatmulNext = document.getElementById('btn-matmul-next');
  if (btnMatmulNext) {
    btnMatmulNext.addEventListener('click', () => modMatMul.nextStep());
  }

  const btnMatmulPrev = document.getElementById('btn-matmul-prev');
  if (btnMatmulPrev) {
    btnMatmulPrev.addEventListener('click', () => modMatMul.prevStep());
  }

  const btnMatmulReset = document.getElementById('btn-matmul-reset');
  if (btnMatmulReset) {
    btnMatmulReset.addEventListener('click', () => modMatMul.reset());
  }

  const btnMatmulRandom = document.getElementById('btn-matmul-random');
  if (btnMatmulRandom) {
    btnMatmulRandom.addEventListener('click', () => modMatMul.randomize());
  }

  document.querySelectorAll('.matmul-dim-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.matmul-dim-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const m = parseInt(btn.dataset.m);
      const k = parseInt(btn.dataset.k);
      const n = parseInt(btn.dataset.n);
      modMatMul.setDimensions(m, k, n);
      document.getElementById('matmul-dim-a').textContent = `${m}×${k}`;
      document.getElementById('matmul-dim-b').textContent = `${k}×${n}`;
      document.getElementById('matmul-dim-c').textContent = `${m}×${n}`;
    });
  });

  // Initial render of MatMul
  modMatMul.render();

  // ==========================================
  // MODULE: VECTOR GEOMETRY CONTROLS
  // ==========================================
  document.querySelectorAll('[data-vmode]').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('[data-vmode]').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const mode = btn.dataset.vmode;
      const scalarsBox = document.getElementById('vec-scalars-box');
      if (scalarsBox) {
        scalarsBox.style.display = mode === 'linear_combination' ? 'block' : 'none';
      }
      modVectors.setMode(mode);
    });
  });

  const updateScalars = () => {
    const c1 = parseFloat(document.getElementById('vec-c1-slider').value);
    const c2 = parseFloat(document.getElementById('vec-c2-slider').value);
    document.getElementById('vec-c1-val').textContent = c1;
    document.getElementById('vec-c2-val').textContent = c2;
    modVectors.setScalars(c1, c2);
  };

  const c1Slider = document.getElementById('vec-c1-slider');
  const c2Slider = document.getElementById('vec-c2-slider');
  if (c1Slider) c1Slider.addEventListener('input', updateScalars);
  if (c2Slider) c2Slider.addEventListener('input', updateScalars);

  const updateVectorsFromInputs = () => {
    const ux = parseFloat(document.getElementById('vec-ux').value) || 0;
    const uy = parseFloat(document.getElementById('vec-uy').value) || 0;
    const vx = parseFloat(document.getElementById('vec-vx').value) || 0;
    const vy = parseFloat(document.getElementById('vec-vy').value) || 0;
    modVectors.u = [ux, uy];
    modVectors.v = [vx, vy];
    modVectors.updateHandles();
    modVectors.render();
    modVectors.updateMathCard();
  };

  ['vec-ux', 'vec-uy', 'vec-vx', 'vec-vy'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateVectorsFromInputs);
  });

  // ==========================================
  // MODULE 1 (TRANSFORMATIONS) CONTROLS
  // ==========================================
  const updateMatrixFromInputs = () => {
    const w11 = parseFloat(document.getElementById('m11').value) || 0;
    const w12 = parseFloat(document.getElementById('m12').value) || 0;
    const w21 = parseFloat(document.getElementById('m21').value) || 0;
    const w22 = parseFloat(document.getElementById('m22').value) || 0;
    modTransform.setMatrix(w11, w12, w21, w22);
  };

  const updateBiasFromInputs = () => {
    const b1 = parseFloat(document.getElementById('b1').value) || 0;
    const b2 = parseFloat(document.getElementById('b2').value) || 0;
    modTransform.setBias(b1, b2);
  };

  ['m11', 'm12', 'm21', 'm22'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateMatrixFromInputs);
  });
  ['b1', 'b2'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateBiasFromInputs);
  });

  // Presets
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const type = btn.dataset.preset;
      if (type === 'identity') {
        modTransform.animateTransition([[1, 0], [0, 1]]);
      } else if (type === 'rotate45') {
        const rad = Math.PI / 4;
        modTransform.animateTransition([[Math.cos(rad), -Math.sin(rad)], [Math.sin(rad), Math.cos(rad)]]);
      } else if (type === 'scale') {
        modTransform.animateTransition([[2.0, 0], [0, 0.6]]);
      } else if (type === 'shear') {
        modTransform.animateTransition([[1, 1.2], [0, 1]]);
      } else if (type === 'reflection') {
        modTransform.animateTransition([[1, 0], [0, -1]]);
      } else if (type === 'singular') {
        modTransform.animateTransition([[1, 1], [1, 1]]);
      }
    });
  });

  // Invariant Eigenvectors Toggle
  const eigenToggle = document.getElementById('toggle-eigenvectors');
  if (eigenToggle) {
    eigenToggle.addEventListener('change', (e) => {
      modTransform.showEigenvectors = e.target.checked;
      modTransform.render();
      modTransform.updateMathCard();
    });
  }

  // Neural Layer Toggle
  const neuralToggle = document.getElementById('toggle-neural');
  if (neuralToggle) {
    neuralToggle.addEventListener('change', (e) => {
      modTransform.showNeuralLayer = e.target.checked;
      modTransform.render();
    });
  }

  const actSelect = document.getElementById('select-activation');
  if (actSelect) {
    actSelect.addEventListener('change', (e) => {
      modTransform.activation = e.target.value;
      modTransform.render();
    });
  }

  const btnResetView1 = document.getElementById('btn-reset-view-1');
  if (btnResetView1) {
    btnResetView1.addEventListener('click', () => {
      canvas1.resetView();
      modTransform.render();
    });
  }

  // ==========================================
  // MODULE 2 (PCA 2D & 3D) CONTROLS
  // ==========================================
  const pcaSlider = document.getElementById('pca-projection-slider');
  if (pcaSlider) {
    pcaSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      document.getElementById('pca-proj-val').textContent = `${Math.round(val * 100)}%`;
      modPCA.setProjectionRatio(val);
    });
  }

  const btnPca2D = document.getElementById('btn-pca-mode-2d');
  const btnPca3D = document.getElementById('btn-pca-mode-3d');
  if (btnPca2D && btnPca3D) {
    btnPca2D.addEventListener('click', () => {
      btnPca2D.classList.add('active');
      btnPca3D.classList.remove('active');
      document.getElementById('pca-footer-hint').textContent = 'Кликните на холст, чтобы добавить новые точки';
      modPCA.setMode('2d');
    });
    btnPca3D.addEventListener('click', () => {
      btnPca3D.classList.add('active');
      btnPca2D.classList.remove('active');
      document.getElementById('pca-footer-hint').textContent = '🖱️ Вращайте 3D сцену зажатой мышкой, двигайте слайдер проекции';
      modPCA.setMode('3d');
    });
  }

  document.querySelectorAll('.pca-gen-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      if (modPCA.mode === '2d') {
        modPCA.generateSampleData(btn.dataset.type);
      } else {
        modPCA.generateSampleData3D();
      }
    });
  });

  const btnResetView2 = document.getElementById('btn-reset-view-2');
  if (btnResetView2) {
    btnResetView2.addEventListener('click', () => {
      canvas2.resetView();
      modPCA.render();
    });
  }

  // ==========================================
  // MODULE 3 (LEAST SQUARES) CONTROLS
  // ==========================================
  const btnStepGD = document.getElementById('btn-step-gd');
  if (btnStepGD) {
    btnStepGD.addEventListener('click', () => {
      modLS.runGradientDescentStep();
    });
  }

  const btnResetLS = document.getElementById('btn-reset-ls');
  if (btnResetLS) {
    btnResetLS.addEventListener('click', () => {
      modLS.solveOLS();
    });
  }

  const btnScrambleLS = document.getElementById('btn-scramble-ls');
  if (btnScrambleLS) {
    btnScrambleLS.addEventListener('click', () => {
      modLS.scrambleWeights();
    });
  }

  const btnAutoGD = document.getElementById('btn-auto-gd');
  if (btnAutoGD) {
    btnAutoGD.addEventListener('click', () => {
      modLS.runAutoGD();
    });
  }

  const btnResetView3 = document.getElementById('btn-reset-view-3');
  if (btnResetView3) {
    btnResetView3.addEventListener('click', () => {
      canvas3.resetView();
      modLS.render();
    });
  }

  // Initial call to display LS status
  modLS.updateStatusUI();

  // ==========================================
  // MODULE 4 (TRAINER) CONTROLS
  // ==========================================
  const selectChallenge = document.getElementById('select-challenge');
  if (selectChallenge) {
    selectChallenge.addEventListener('change', (e) => {
      modTrainer.loadChallenge(e.target.value);
    });
    modTrainer.loadChallenge(selectChallenge.value);
  }

  const updateTrainerMatrix = () => {
    const w11 = parseFloat(document.getElementById('tr_m11').value) || 0;
    const w12 = parseFloat(document.getElementById('tr_m12').value) || 0;
    const w21 = parseFloat(document.getElementById('tr_m21').value) || 0;
    const w22 = parseFloat(document.getElementById('tr_m22').value) || 0;
    modTrainer.userMatrix = [[w11, w12], [w21, w22]];
    modTrainer.render();
  };

  ['tr_m11', 'tr_m12', 'tr_m21', 'tr_m22'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateTrainerMatrix);
  });

  const angleSlider = document.getElementById('tr-angle-slider');
  if (angleSlider) {
    angleSlider.addEventListener('input', (e) => {
      modTrainer.userAngle = parseFloat(e.target.value);
      document.getElementById('tr-angle-val').textContent = `${modTrainer.userAngle}°`;
      modTrainer.render();
    });
  }

  const rankSlider = document.getElementById('tr-rank-slider');
  if (rankSlider) {
    rankSlider.addEventListener('input', (e) => {
      modTrainer.userRank = parseInt(e.target.value, 10);
      const valEl = document.getElementById('tr-rank-val');
      if (valEl) valEl.textContent = `k = ${modTrainer.userRank}`;
      modTrainer.render();
    });
  }

  const updateTrainerSVM = () => {
    modTrainer.userSVM.w1 = parseFloat(document.getElementById('tr_w1').value) || 0;
    modTrainer.userSVM.w2 = parseFloat(document.getElementById('tr_w2').value) || 0;
    modTrainer.userSVM.b = parseFloat(document.getElementById('tr_b').value) || 0;
    modTrainer.render();
  };
  ['tr_w1', 'tr_w2', 'tr_b'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateTrainerSVM);
  });

  const updateTrainerLS = () => {
    const s = parseFloat(document.getElementById('tr_slope').value) || 0;
    const b = parseFloat(document.getElementById('tr_intercept').value) || 0;
    modTrainer.userLS.slope = s;
    modTrainer.userLS.intercept = b;
    const sVal = document.getElementById('tr-slope-val');
    const bVal = document.getElementById('tr-intercept-val');
    if (sVal) sVal.textContent = `k = ${s.toFixed(2)}`;
    if (bVal) bVal.textContent = `b = ${b.toFixed(2)}`;
    modTrainer.render();
  };
  ['tr_slope', 'tr_intercept'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateTrainerLS);
  });

  const btnVerify = document.getElementById('btn-verify-challenge');
  if (btnVerify) {
    btnVerify.addEventListener('click', () => {
      modTrainer.verifySubmission();
    });
  }

  // ==========================================
  // SVD MODULE EVENT CONTROLLERS
  // ==========================================
  const btnSvdModeGeom = document.getElementById('btn-svd-mode-geom');
  const btnSvdModeLora = document.getElementById('btn-svd-mode-lora');
  const svdGeomControls = document.getElementById('svd-geom-controls');
  const svdLoraControls = document.getElementById('svd-lora-controls');
  const svdLegendGeom = document.getElementById('svd-legend-geom');
  const svdLegendLora = document.getElementById('svd-legend-lora');
  const svdFooterHint = document.getElementById('svd-footer-hint');

  if (btnSvdModeGeom && btnSvdModeLora) {
    btnSvdModeGeom.addEventListener('click', () => {
      btnSvdModeGeom.classList.add('active');
      btnSvdModeLora.classList.remove('active');
      if (svdGeomControls) svdGeomControls.style.display = 'block';
      if (svdLoraControls) svdLoraControls.style.display = 'none';
      if (svdLegendGeom) svdLegendGeom.style.display = 'flex';
      if (svdLegendLora) svdLegendLora.style.display = 'none';
      if (svdFooterHint) svdFooterHint.textContent = 'Двигайте слайдер этапов: x → Vᵀx → ΣVᵀx → UΣVᵀx';
      modSVD.mode = 'geom';
      modSVD.render();
      modSVD.updateMathCard();
    });

    btnSvdModeLora.addEventListener('click', () => {
      btnSvdModeLora.classList.add('active');
      btnSvdModeGeom.classList.remove('active');
      if (svdGeomControls) svdGeomControls.style.display = 'none';
      if (svdLoraControls) svdLoraControls.style.display = 'block';
      if (svdLegendGeom) svdLegendGeom.style.display = 'none';
      if (svdLegendLora) svdLegendLora.style.display = 'flex';
      if (svdFooterHint) svdFooterHint.textContent = 'LoRA: матрица A_k формируется суммой k внешних произведений σᵢ uᵢ vᵢᵀ';
      modSVD.mode = 'lora';
      modSVD.computeLoRA();
      modSVD.render();
      modSVD.updateGrids();
    });
  }

  const svdStageSlider = document.getElementById('svd-stage-slider');
  if (svdStageSlider) {
    svdStageSlider.addEventListener('input', (e) => {
      const s = parseFloat(e.target.value);
      modSVD.stage = s;
      const valEl = document.getElementById('svd-stage-val');
      if (valEl) {
        valEl.textContent = s < 0.9 ? `Этап 1: Vᵀ (${s.toFixed(2)})` :
                            s < 1.9 ? `Этап 2: Σ Vᵀ (${s.toFixed(2)})` :
                            s < 2.9 ? `Этап 3: U Σ Vᵀ (${s.toFixed(2)})` :
                                      'Финал: A x = U Σ Vᵀ x';
      }
      modSVD.render();
    });
  }

  const updateSVDMatrix = () => {
    const a11 = parseFloat(document.getElementById('svd-a11').value) || 0;
    const a12 = parseFloat(document.getElementById('svd-a12').value) || 0;
    const a21 = parseFloat(document.getElementById('svd-a21').value) || 0;
    const a22 = parseFloat(document.getElementById('svd-a22').value) || 0;
    modSVD.A = [[a11, a12], [a21, a22]];
    modSVD.compute2DSVD();
    modSVD.render();
    modSVD.updateMathCard();
  };

  ['svd-a11', 'svd-a12', 'svd-a21', 'svd-a22'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateSVDMatrix);
  });

  document.querySelectorAll('.svd-preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.svd-preset-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      modSVD.setPreset(btn.dataset.preset);
    });
  });

  const svdRankSlider = document.getElementById('svd-rank-slider');
  if (svdRankSlider) {
    svdRankSlider.addEventListener('input', (e) => {
      const r = parseInt(e.target.value, 10);
      modSVD.rank = r;
      const valEl = document.getElementById('svd-rank-val');
      if (valEl) valEl.textContent = `k = ${r}`;
      modSVD.computeLoRA();
      modSVD.render();
    });
  }

  document.querySelectorAll('.svd-pattern-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.svd-pattern-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      modSVD.setPattern(btn.dataset.pattern);
    });
  });

  // ==========================================
  // EMBEDDINGS & RAG MODULE EVENT CONTROLLERS
  // ==========================================
  const btnEmbModeRag = document.getElementById('btn-emb-mode-rag');
  const btnEmbModeAnalogy = document.getElementById('btn-emb-mode-analogy');
  const embRagControls = document.getElementById('emb-rag-controls');
  const embAnalogyControls = document.getElementById('emb-analogy-controls');
  const embLegendRag = document.getElementById('emb-legend-rag');
  const embLegendAnalogy = document.getElementById('emb-legend-analogy');

  if (btnEmbModeRag && btnEmbModeAnalogy) {
    btnEmbModeRag.addEventListener('click', () => {
      btnEmbModeRag.classList.add('active');
      btnEmbModeAnalogy.classList.remove('active');
      if (embRagControls) embRagControls.style.display = 'block';
      if (embAnalogyControls) embAnalogyControls.style.display = 'none';
      if (embLegendRag) embLegendRag.style.display = 'flex';
      if (embLegendAnalogy) embLegendAnalogy.style.display = 'none';
      modEmbeddings.setMode('rag');
    });

    btnEmbModeAnalogy.addEventListener('click', () => {
      btnEmbModeAnalogy.classList.add('active');
      btnEmbModeRag.classList.remove('active');
      if (embRagControls) embRagControls.style.display = 'none';
      if (embAnalogyControls) embAnalogyControls.style.display = 'block';
      if (embLegendRag) embLegendRag.style.display = 'none';
      if (embLegendAnalogy) embLegendAnalogy.style.display = 'flex';
      modEmbeddings.setMode('analogy');
    });
  }

  const embTopKSlider = document.getElementById('emb-topk-slider');
  if (embTopKSlider) {
    embTopKSlider.addEventListener('input', (e) => {
      const k = parseInt(e.target.value, 10);
      modEmbeddings.topK = k;
      const valEl = document.getElementById('emb-topk-val');
      if (valEl) valEl.textContent = `K = ${k}`;
      modEmbeddings.render();
      modEmbeddings.updateRAGTable();
    });
  }

  const embQuerySlider = document.getElementById('emb-query-angle-slider');
  if (embQuerySlider) {
    embQuerySlider.addEventListener('input', (e) => {
      const angle = parseFloat(e.target.value);
      modEmbeddings.queryAngle = angle;
      const valEl = document.getElementById('emb-query-angle-val');
      if (valEl) valEl.textContent = `${angle}°`;
      modEmbeddings.render();
      modEmbeddings.updateRAGTable();
    });
  }

  const btnAddConcept = document.getElementById('btn-emb-add-concept');
  const inputConceptLabel = document.getElementById('emb-new-label');
  if (btnAddConcept && inputConceptLabel) {
    const doAdd = () => {
      const txt = inputConceptLabel.value.trim();
      if (txt) {
        modEmbeddings.addCustomConcept(txt);
        inputConceptLabel.value = '';
      }
    };
    btnAddConcept.addEventListener('click', doAdd);
    inputConceptLabel.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') doAdd();
    });
  }

  document.querySelectorAll('.emb-analogy-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.emb-analogy-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      modEmbeddings.setAnalogy(btn.dataset.analogy);
    });
  });

  const btnEmbAnalogyPlay = document.getElementById('btn-emb-analogy-play');
  if (btnEmbAnalogyPlay) {
    btnEmbAnalogyPlay.addEventListener('click', () => {
      modEmbeddings.togglePlayAnalogy();
    });
  }

  const btnEmbAnalogyReset = document.getElementById('btn-emb-analogy-reset');
  if (btnEmbAnalogyReset) {
    btnEmbAnalogyReset.addEventListener('click', () => {
      modEmbeddings.resetAnalogyVectors();
    });
  }

  const btnResetViewSVD = document.getElementById('btn-reset-view-svd');
  if (btnResetViewSVD) {
    btnResetViewSVD.addEventListener('click', () => {
      canvasSVD.resetView();
      modSVD.render();
    });
  }

  const btnResetViewEmb = document.getElementById('btn-reset-view-emb');
  if (btnResetViewEmb) {
    btnResetViewEmb.addEventListener('click', () => {
      canvasEmbeddings.resetView();
      modEmbeddings.render();
    });
  }

  // Initial table render
  modEmbeddings.updateRAGTable();
  modEmbeddings.updateAnalogyUI();

  // ==========================================
  // MODULE: SVM & LINEAR CLASSIFICATION CONTROLS
  // ==========================================
  const updateSVMFromSliders = () => {
    const w1 = parseFloat(document.getElementById('svm-w1-slider').value) || 0;
    const w2 = parseFloat(document.getElementById('svm-w2-slider').value) || 0;
    const b = parseFloat(document.getElementById('svm-b-slider').value) || 0;
    modSVM.w = [w1, w2];
    modSVM.b = b;
    const w1Val = document.getElementById('svm-w1-val');
    const w2Val = document.getElementById('svm-w2-val');
    const bVal = document.getElementById('svm-b-val');
    if (w1Val) w1Val.textContent = w1;
    if (w2Val) w2Val.textContent = w2;
    if (bVal) bVal.textContent = b;
    modSVM.computeAndRender(false);
  };

  ['svm-w1-slider', 'svm-w2-slider', 'svm-b-slider'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateSVMFromSliders);
  });

  const btnSvmClassPos = document.getElementById('btn-svm-class-pos');
  const btnSvmClassNeg = document.getElementById('btn-svm-class-neg');
  if (btnSvmClassPos && btnSvmClassNeg) {
    btnSvmClassPos.addEventListener('click', () => {
      modSVM.activeClass = 1;
      btnSvmClassPos.classList.add('active');
      btnSvmClassNeg.classList.remove('active');
    });
    btnSvmClassNeg.addEventListener('click', () => {
      modSVM.activeClass = -1;
      btnSvmClassNeg.classList.add('active');
      btnSvmClassPos.classList.remove('active');
    });
  }

  const toggleMargin = document.getElementById('svm-toggle-margin');
  if (toggleMargin) {
    toggleMargin.addEventListener('change', (e) => {
      modSVM.showMargin = e.target.checked;
      modSVM.render();
    });
  }

  const toggleSV = document.getElementById('svm-toggle-sv');
  if (toggleSV) {
    toggleSV.addEventListener('change', (e) => {
      modSVM.showSupportVectors = e.target.checked;
      modSVM.render();
    });
  }

  const toggleRegions = document.getElementById('svm-toggle-regions');
  if (toggleRegions) {
    toggleRegions.addEventListener('change', (e) => {
      modSVM.showDecisionRegions = e.target.checked;
      modSVM.render();
    });
  }

  const btnSvmResetPoints = document.getElementById('btn-svm-reset-points');
  if (btnSvmResetPoints) {
    btnSvmResetPoints.addEventListener('click', () => {
      modSVM.loadDefaultPoints();
      modSVM.computeAndRender(true);
    });
  }

  const btnResetViewSVM = document.getElementById('btn-reset-view-svm');
  if (btnResetViewSVM) {
    btnResetViewSVM.addEventListener('click', () => {
      canvasSVM.resetView();
      modSVM.render();
    });
  }

  const btnUndoSVM = document.getElementById('btn-undo-svm');
  if (btnUndoSVM) btnUndoSVM.addEventListener('click', () => modSVM.undo());
  const btnRedoSVM = document.getElementById('btn-redo-svm');
  if (btnRedoSVM) btnRedoSVM.addEventListener('click', () => modSVM.redo());

  // ==========================================
  // MODULE: SELF-ATTENTION IN TRANSFORMERS CONTROLS
  // ==========================================
  const attnTempSlider = document.getElementById('attn-temp-slider');
  if (attnTempSlider) {
    attnTempSlider.addEventListener('input', (e) => {
      const t = parseFloat(e.target.value);
      modAttention.setTemperature(t);
      const valEl = document.getElementById('attn-temp-val');
      if (valEl) valEl.textContent = `${t.toFixed(1)}×`;
    });
  }

  const btnAttnRandomize = document.getElementById('btn-attn-randomize');
  if (btnAttnRandomize) {
    btnAttnRandomize.addEventListener('click', () => {
      modAttention.randomize();
    });
  }

  const btnResetViewAttn = document.getElementById('btn-reset-view-attn');
  if (btnResetViewAttn) {
    btnResetViewAttn.addEventListener('click', () => {
      canvasAttention.resetView();
      modAttention.render();
    });
  }

  // ==========================================
  // MODULE: HESSIAN CONDITION NUMBER & LOSS CONTROLS
  // ==========================================
  const updateHessianFromSliders = () => {
    const kappa = parseFloat(document.getElementById('hessian-kappa-slider').value) || 10;
    const angle = parseFloat(document.getElementById('hessian-angle-slider').value) || 25;
    const lr = parseFloat(document.getElementById('hessian-lr-slider').value) || 0.12;
    const beta = parseFloat(document.getElementById('hessian-beta-slider').value) || 0.85;

    const kVal = document.getElementById('hessian-kappa-val');
    const aVal = document.getElementById('hessian-angle-val');
    const lrVal = document.getElementById('hessian-lr-val');
    const bVal = document.getElementById('hessian-beta-val');

    if (kVal) kVal.textContent = kappa.toFixed(1);
    if (aVal) aVal.textContent = `${angle}°`;
    if (lrVal) lrVal.textContent = lr.toFixed(2);
    if (bVal) bVal.textContent = beta.toFixed(2);

    modHessian.setParameters(kappa, angle, lr, beta);
  };

  ['hessian-kappa-slider', 'hessian-angle-slider', 'hessian-lr-slider', 'hessian-beta-slider'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateHessianFromSliders);
  });

  document.querySelectorAll('.hessian-preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const kappa = parseFloat(btn.dataset.kappa);
      const lr = parseFloat(btn.dataset.lr);
      if (document.getElementById('hessian-kappa-slider')) document.getElementById('hessian-kappa-slider').value = kappa;
      if (document.getElementById('hessian-lr-slider')) document.getElementById('hessian-lr-slider').value = lr;
      updateHessianFromSliders();
    });
  });

  const btnHessianResetStart = document.getElementById('btn-hessian-reset-start');
  if (btnHessianResetStart) {
    btnHessianResetStart.addEventListener('click', () => {
      modHessian.startPoint = [-3.0, 2.8];
      modHessian.computeSimulation();
      modHessian.render();
      modHessian.updateMathCard();
    });
  }

  const btnResetViewHessian = document.getElementById('btn-reset-view-hessian');
  if (btnResetViewHessian) {
    btnResetViewHessian.addEventListener('click', () => {
      canvasHessian.resetView();
      modHessian.render();
    });
  }

  // ==========================================
  // UNDO / REDO CONTROLLERS & KEYBOARD SHORTCUTS
  // ==========================================
  const btnUndoPCA = document.getElementById('btn-undo-pca');
  if (btnUndoPCA) btnUndoPCA.addEventListener('click', () => modPCA.undo());
  const btnRedoPCA = document.getElementById('btn-redo-pca');
  if (btnRedoPCA) btnRedoPCA.addEventListener('click', () => modPCA.redo());

  const btnUndoLS = document.getElementById('btn-undo-ls');
  if (btnUndoLS) btnUndoLS.addEventListener('click', () => modLS.undo());
  const btnRedoLS = document.getElementById('btn-redo-ls');
  if (btnRedoLS) btnRedoLS.addEventListener('click', () => modLS.redo());

  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
      e.preventDefault();
      if (e.shiftKey) {
        if (currentActiveTab === 'tab-module2') modPCA.redo();
        else if (currentActiveTab === 'tab-module3') modLS.redo();
        else if (currentActiveTab === 'tab-svm') modSVM.redo();
      } else {
        if (currentActiveTab === 'tab-module2') modPCA.undo();
        else if (currentActiveTab === 'tab-module3') modLS.undo();
        else if (currentActiveTab === 'tab-svm') modSVM.undo();
      }
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
      e.preventDefault();
      if (currentActiveTab === 'tab-module2') modPCA.redo();
      else if (currentActiveTab === 'tab-module3') modLS.redo();
      else if (currentActiveTab === 'tab-svm') modSVM.redo();
    }
  });

  // ==========================================
  // DATASET MANAGER MODAL & FILE UPLOADER
  // ==========================================
  let activeDatasetTargetModule = 'pca';
  const modalDataset = document.getElementById('modal-dataset');
  const btnCloseDatasetModal = document.getElementById('btn-close-dataset-modal');

  const openDatasetModal = (targetModule) => {
    activeDatasetTargetModule = targetModule;
    if (modalDataset) modalDataset.style.display = 'flex';
  };

  const closeDatasetModal = () => {
    if (modalDataset) modalDataset.style.display = 'none';
  };

  if (btnCloseDatasetModal) btnCloseDatasetModal.addEventListener('click', closeDatasetModal);
  if (modalDataset) {
    modalDataset.addEventListener('click', (e) => {
      if (e.target === modalDataset) closeDatasetModal();
    });
  }

  const btnDatasetPCA = document.getElementById('btn-dataset-pca');
  if (btnDatasetPCA) btnDatasetPCA.addEventListener('click', () => openDatasetModal('pca'));

  const btnDatasetLS = document.getElementById('btn-dataset-ls');
  if (btnDatasetLS) btnDatasetLS.addEventListener('click', () => openDatasetModal('ls'));

  const btnDatasetSVM = document.getElementById('btn-dataset-svm');
  if (btnDatasetSVM) btnDatasetSVM.addEventListener('click', () => openDatasetModal('svm'));

  const applyDatasetToModule = (points) => {
    if (!points || points.length === 0) return;
    const normPts = datasetManager.normalizePoints(points);

    if (activeDatasetTargetModule === 'pca') {
      modPCA.points = normPts.map(p => [p.x, p.y]);
      modPCA.pushHistory();
      modPCA.computeAndRender(true);
      switchTab('tab-module2');
    } else if (activeDatasetTargetModule === 'ls') {
      modLS.points = normPts.map(p => [p.x, p.y]);
      modLS.initCanvasHandles();
      modLS.pushHistory();
      modLS.solveOLS();
      modLS.render();
      modLS.updateMathCard();
      modLS.updateStatusUI();
      switchTab('tab-module3');
    } else if (activeDatasetTargetModule === 'svm') {
      modSVM.points = normPts.map(p => ({ x: p.x, y: p.y, label: p.label || 1 }));
      modSVM.pushHistory();
      modSVM.computeAndRender(true);
      switchTab('tab-svm');
    }
    closeDatasetModal();
  };

  document.querySelectorAll('.btn-dataset-choice').forEach(btn => {
    btn.addEventListener('click', () => {
      const dsId = btn.dataset.dataset;
      const ds = datasetManager.getDataset(dsId);
      if (ds) {
        applyDatasetToModule(ds.points);
      }
    });
  });

  // Custom File Uploader & Parser (CSV / JSON)
  const dropzone = document.getElementById('dataset-dropzone');
  const fileInput = document.getElementById('input-dataset-file');
  let parsedCustomPoints = [];

  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        handleDatasetFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleDatasetFile(e.target.files[0]);
      }
    });
  }

  const handleDatasetFile = (file) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target.result;
        let parsed;
        if (file.name.endsWith('.json')) {
          parsed = datasetManager.parseJSON(text);
        } else {
          parsed = datasetManager.parseCSV(text);
        }

        const previewBox = document.getElementById('dataset-preview-box');
        const previewTable = document.getElementById('dataset-preview-table');
        const previewTitle = document.getElementById('dataset-preview-title');

        // Extract points from first two numeric columns
        parsedCustomPoints = [];
        for (const row of parsed.rows) {
          const rawX = typeof row[0] === 'string' ? row[0].replace(',', '.') : row[0];
          const rawY = typeof row[1] === 'string' ? row[1].replace(',', '.') : row[1];
          const x = parseFloat(rawX);
          const y = parseFloat(rawY);
          const rawLabel = row.length > 2 ? (typeof row[2] === 'string' ? row[2].replace(',', '.') : row[2]) : null;
          const label = rawLabel !== null ? (parseFloat(rawLabel) >= 0 ? 1 : -1) : 1;
          if (!isNaN(x) && !isNaN(y)) {
            parsedCustomPoints.push({ x, y, label });
          }
        }

        if (parsedCustomPoints.length === 0) {
          alert("Не удалось извлечь числовые координаты X и Y из первых двух колонок файла.");
          return;
        }

        if (previewTitle) previewTitle.textContent = `Файл: ${file.name} (${parsedCustomPoints.length} точек):`;
        if (previewTable) {
          let html = `<table class="neo-table"><thead><tr><th>#</th><th>${parsed.headers[0] || 'X'}</th><th>${parsed.headers[1] || 'Y'}</th></tr></thead><tbody>`;
          parsed.rows.slice(0, 5).forEach((r, idx) => {
            html += `<tr><td>${idx + 1}</td><td>${r[0]}</td><td>${r[1]}</td></tr>`;
          });
          html += '</tbody></table>';
          previewTable.innerHTML = html;
        }
        if (previewBox) previewBox.style.display = 'block';
      } catch (err) {
        alert("Ошибка чтения файла данных: " + err.message);
      }
    };
    reader.readAsText(file);
  };

  const btnApplyCustom = document.getElementById('btn-apply-custom-dataset');
  if (btnApplyCustom) {
    btnApplyCustom.addEventListener('click', () => {
      if (parsedCustomPoints.length > 0) {
        applyDatasetToModule(parsedCustomPoints);
      }
    });
  }

  // ==========================================
  // REPORT EXPORT CONTROLLER (PDF / PRINT)
  // ==========================================
  const btnOpenReport = document.getElementById('btn-open-report');
  if (btnOpenReport) {
    btnOpenReport.addEventListener('click', () => {
      reportGen.openModal();
    });
  }

  // ==========================================
  // GLOBAL SITE DESIGN & THEME MODE CONTROLLER
  // Styles: 'neobrutalism' | 'retro-studio'
  // Modes:  'light' | 'dark'
  // ==========================================
  const allCanvases = [canvasVectors, canvas1, canvas2, canvas3, canvas4, canvasSVD, canvasEmbeddings, canvasSVM, canvasAttention, canvasHessian];
  const allModules = [modMatMul, modVectors, modTransform, modPCA, modLS, modTrainer, modSVD, modEmbeddings, modSVM, modAttention, modHessian];

  let currentStyle = 'neobrutalism';
  let currentMode = 'light';

  const applyStyleAndMode = (styleName, modeName, save = true) => {
    currentStyle = styleName === 'retro-studio' ? 'retro-studio' : 'neobrutalism';
    currentMode = modeName === 'dark' ? 'dark' : 'light';

    document.body.dataset.theme = currentStyle;
    document.body.dataset.style = currentStyle;
    document.body.dataset.mode = currentMode;

    // Update active state on style buttons
    document.querySelectorAll('.theme-toggle-btn').forEach(btn => {
      if (btn.dataset.styleTarget === currentStyle) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    // Update active state on mode buttons
    document.querySelectorAll('.mode-toggle-btn').forEach(btn => {
      if (btn.dataset.modeTarget === currentMode) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    // Update canvases coordinate themes & mode
    allCanvases.forEach(c => {
      if (c && typeof c.setTheme === 'function') {
        c.setTheme(currentStyle, currentMode);
      }
    });

    // Update modules visual palette
    const targetPalette = currentStyle === 'retro-studio' ? 2 : 1;
    allModules.forEach(m => {
      if (m && typeof m.setPalette === 'function') {
        m.setPalette(targetPalette, currentMode);
      }
    });

    // Re-render currently visible modules & canvases
    modMatMul.render();
    modVectors.render();
    modTransform.render();
    modPCA.render();
    modLS.render();
    modTrainer.render();
    modSVD.render();
    modEmbeddings.render();
    modSVM.render();
    modAttention.render();
    modHessian.render();

    if (save) {
      try {
        localStorage.setItem('laml_style', currentStyle);
        localStorage.setItem('laml_mode', currentMode);
      } catch (e) {
        /* ignore storage error */
      }
    }
  };

  document.querySelectorAll('.theme-toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetStyle = btn.dataset.styleTarget;
      if (targetStyle) {
        applyStyleAndMode(targetStyle, currentMode, true);
      }
    });
  });

  document.querySelectorAll('.mode-toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const targetMode = btn.dataset.modeTarget;
      if (targetMode) {
        applyStyleAndMode(currentStyle, targetMode, true);
      }
    });
  });

  // Restore saved style and mode or default to neobrutalism + light
  try {
    const savedStyle = localStorage.getItem('laml_style') || localStorage.getItem('laml_site_theme') || 'neobrutalism';
    const savedMode = localStorage.getItem('laml_mode') || 'light';
    applyStyleAndMode(savedStyle, savedMode, false);
  } catch (e) {
    applyStyleAndMode('neobrutalism', 'light', false);
  }
});

