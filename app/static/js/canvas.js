/**
 * InteractiveCanvas: 2D Cartesian Coordinate & Vector Graphics Engine
 * Supports pan, zoom, retina scaling, transformed grids, vectors with arrowheads,
 * draggable handles, and smooth animation loops.
 */

class InteractiveCanvas {
  constructor(canvasElement, options = {}) {
    this.canvas = canvasElement;
    this.ctx = canvasElement.getContext('2d');
    
    // Virtual Cartesian Coordinate System settings
    this.scale = options.scale || 65; // pixels per 1.0 math unit
    this.originX = this.canvas.width / 2;
    this.originY = this.canvas.height / 2;
    
    this.minScale = 15;
    this.maxScale = 250;
    
    // Interactive handles (draggable points)
    this.handles = [];
    this.activeHandle = null;
    this.isPanning = false;
    this.panStartX = 0;
    this.panStartY = 0;

    // Callbacks
    this.onHandleDrag = options.onHandleDrag || null;
    // Theme styling ('neobrutalism' | 'retro-studio') & Mode ('light' | 'dark')
    this.theme = options.theme || 'neobrutalism';
    this.mode = options.mode || 'light';

    this.initHiDPI();
    this.initEvents();
  }

  setTheme(theme, mode = 'light') {
    this.theme = theme;
    this.mode = mode;
  }

  initHiDPI() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    
    const displayWidth = rect.width || this.canvas.parentElement.clientWidth || 700;
    const displayHeight = rect.height || this.canvas.parentElement.clientHeight || 520;

    this.canvas.width = displayWidth * dpr;
    this.canvas.height = displayHeight * dpr;
    this.ctx.scale(dpr, dpr);

    this.displayWidth = displayWidth;
    this.displayHeight = displayHeight;
    this.originX = displayWidth / 2;
    this.originY = displayHeight / 2;
  }

  get width() {
    return this.displayWidth || (this.canvas ? this.canvas.clientWidth : 700) || 700;
  }

  get height() {
    return this.displayHeight || (this.canvas ? this.canvas.clientHeight : 520) || 520;
  }

  resize() {
    this.initHiDPI();
  }

  resetView() {
    this.scale = 65;
    this.originX = this.displayWidth / 2;
    this.originY = this.displayHeight / 2;
  }

  // Coordinate transforms
  mathToCanvas(x, y) {
    return {
      px: this.originX + x * this.scale,
      py: this.originY - y * this.scale
    };
  }

  canvasToMath(px, py) {
    return {
      x: (px - this.originX) / this.scale,
      y: (this.originY - py) / this.scale
    };
  }

  addHandle(id, x, y, radius = 9, color = '#38bdf8', label = '') {
    const existing = this.handles.find(h => h.id === id);
    if (existing) {
      existing.x = x;
      existing.y = y;
      existing.color = color;
      existing.label = label;
    } else {
      this.handles.push({ id, x, y, radius, color, label });
    }
  }

  getHandle(id) {
    return this.handles.find(h => h.id === id);
  }

  clearHandles() {
    this.handles = [];
  }

  initEvents() {
    const getPos = (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const clientX = e.touches ? e.touches[0].clientX : e.clientX;
      const clientY = e.touches ? e.touches[0].clientY : e.clientY;
      return {
        px: clientX - rect.left,
        py: clientY - rect.top
      };
    };

    const handleDown = (e) => {
      const { px, py } = getPos(e);
      // Check if clicked on a handle
      for (const h of this.handles) {
        const { px: hx, py: hy } = this.mathToCanvas(h.x, h.y);
        const dist = Math.hypot(px - hx, py - hy);
        if (dist <= h.radius + 6) {
          this.activeHandle = h;
          this.canvas.style.cursor = 'grabbing';
          return;
        }
      }

      // If space or right mouse or empty click -> pan
      if (e.button === 2 || e.button === 1 || e.shiftKey || e.altKey) {
        this.isPanning = true;
        this.panStartX = px - this.originX;
        this.panStartY = py - this.originY;
        this.canvas.style.cursor = 'grabbing';
      } else if (this.onClick) {
        const mathCoord = this.canvasToMath(px, py);
        this.onClick(mathCoord.x, mathCoord.y, e);
      }
    };

    const handleMove = (e) => {
      const { px, py } = getPos(e);

      if (this.activeHandle) {
        const mathCoord = this.canvasToMath(px, py);
        this.activeHandle.x = mathCoord.x;
        this.activeHandle.y = mathCoord.y;
        if (this.onHandleDrag) {
          this.onHandleDrag(this.activeHandle.id, mathCoord.x, mathCoord.y);
        }
      } else if (this.isPanning) {
        this.originX = px - this.panStartX;
        this.originY = py - this.panStartY;
        if (this.onPan) this.onPan();
      } else {
        // Hover cursor check
        let hovering = false;
        for (const h of this.handles) {
          const { px: hx, py: hy } = this.mathToCanvas(h.x, h.y);
          if (Math.hypot(px - hx, py - hy) <= h.radius + 6) {
            hovering = true;
            break;
          }
        }
        this.canvas.style.cursor = hovering ? 'grab' : 'crosshair';
      }
    };

    const handleUp = () => {
      this.activeHandle = null;
      this.isPanning = false;
      this.canvas.style.cursor = 'crosshair';
    };

    this.canvas.addEventListener('mousedown', handleDown);
    window.addEventListener('mousemove', handleMove);
    window.addEventListener('mouseup', handleUp);

    this.canvas.addEventListener('touchstart', (e) => { e.preventDefault(); handleDown(e); }, { passive: false });
    window.addEventListener('touchmove', (e) => { handleMove(e); });
    window.addEventListener('touchend', handleUp);

    // Zoom on wheel
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const { px, py } = getPos(e);
      const mathBefore = this.canvasToMath(px, py);

      const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89;
      const newScale = Math.min(Math.max(this.scale * zoomFactor, this.minScale), this.maxScale);
      this.scale = newScale;

      // Adjust origin to keep mouse position invariant
      this.originX = px - mathBefore.x * this.scale;
      this.originY = py + mathBefore.y * this.scale;
      if (this.onZoom) this.onZoom();
    }, { passive: false });

    // Prevent context menu on canvas
    this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  clear() {
    this.ctx.clearRect(0, 0, this.displayWidth, this.displayHeight);
  }

  // Draw background Cartesian grid
  drawGrid() {
    const ctx = this.ctx;
    const w = this.displayWidth;
    const h = this.displayHeight;

    ctx.save();

    let bgFill = '#ffffff';
    let gridStroke = '#e2e8f0';
    let axisStroke = '#000000';
    let tickColor = '#64748b';

    const isDark = this.mode === 'dark';

    if (this.theme === 'retro-studio') {
      if (isDark) {
        bgFill = '#24151f';
        gridStroke = '#3d2031';
        axisStroke = '#9d687f';
        tickColor = '#d4a5b8';
      } else {
        bgFill = '#fdfbf7';
        gridStroke = '#ecd9df';
        axisStroke = '#4a2835';
        tickColor = '#8b5b6f';
      }
    } else {
      // neobrutalism
      if (isDark) {
        bgFill = '#18181b';
        gridStroke = '#27272a';
        axisStroke = '#71717a';
        tickColor = '#a1a1aa';
      } else {
        bgFill = '#ffffff';
        gridStroke = '#e2e8f0';
        axisStroke = '#000000';
        tickColor = '#64748b';
      }
    }

    ctx.fillStyle = bgFill;
    ctx.fillRect(0, 0, w, h);

    // Determine grid step
    let step = 1;
    if (this.scale < 30) step = 2;
    if (this.scale < 18) step = 5;
    if (this.scale > 120) step = 0.5;

    const leftMath = this.canvasToMath(0, h);
    const rightMath = this.canvasToMath(w, 0);

    const xStart = Math.floor(leftMath.x / step) * step;
    const xEnd = Math.ceil(rightMath.x / step) * step;
    const yStart = Math.floor(leftMath.y / step) * step;
    const yEnd = Math.ceil(rightMath.y / step) * step;

    // Grid lines
    ctx.lineWidth = 1;
    ctx.strokeStyle = gridStroke;

    ctx.beginPath();
    for (let x = xStart; x <= xEnd; x += step) {
      if (Math.abs(x) < 1e-6) continue;
      const { px } = this.mathToCanvas(x, 0);
      ctx.moveTo(px, 0);
      ctx.lineTo(px, h);
    }
    for (let y = yStart; y <= yEnd; y += step) {
      if (Math.abs(y) < 1e-6) continue;
      const { py } = this.mathToCanvas(0, y);
      ctx.moveTo(0, py);
      ctx.lineTo(w, py);
    }
    ctx.stroke();

    // Primary Axes (X and Y)
    ctx.lineWidth = 2;
    ctx.strokeStyle = axisStroke;
    ctx.beginPath();
    // X Axis
    const { py: zeroY } = this.mathToCanvas(0, 0);
    ctx.moveTo(0, zeroY);
    ctx.lineTo(w, zeroY);
    // Y Axis
    const { px: zeroX } = this.mathToCanvas(0, 0);
    ctx.moveTo(zeroX, 0);
    ctx.lineTo(zeroX, h);
    ctx.stroke();

    // Numbers / Ticks
    ctx.fillStyle = tickColor;
    ctx.font = '600 11px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    for (let x = xStart; x <= xEnd; x += step) {
      if (Math.abs(x) < 1e-6) continue;
      const { px, py } = this.mathToCanvas(x, 0);
      ctx.fillText(Number(x.toFixed(1)), px, Math.min(Math.max(py + 6, 8), h - 18));
    }

    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let y = yStart; y <= yEnd; y += step) {
      if (Math.abs(y) < 1e-6) continue;
      const { px, py } = this.mathToCanvas(0, y);
      ctx.fillText(Number(y.toFixed(1)), Math.min(Math.max(px - 8, 24), w - 8), py);
    }
    ctx.restore();
  }

  // Draw Deformed Coordinate Grid (Matrix Transformation)
  drawTransformedGrid(matrix, bias = [0, 0], range = 8) {
    const ctx = this.ctx;
    const [w11, w12] = matrix[0];
    const [w21, w22] = matrix[1];
    const [b1, b2] = bias;

    const isDark = this.mode === 'dark';
    const isRetro = this.theme === 'retro-studio';
    let gridColor = 'rgba(2, 132, 199, 0.22)';
    if (isRetro) {
      gridColor = isDark ? 'rgba(52, 211, 153, 0.28)' : 'rgba(5, 150, 105, 0.22)';
    } else {
      gridColor = isDark ? 'rgba(56, 189, 248, 0.28)' : 'rgba(2, 132, 199, 0.22)';
    }

    ctx.save();
    ctx.lineWidth = 1;
    ctx.strokeStyle = gridColor;

    for (let i = -range; i <= range; i++) {
      // Lines where x = i (varying y from -range to range)
      const p1 = this.mathToCanvas(w11 * i + w12 * (-range) + b1, w21 * i + w22 * (-range) + b2);
      const p2 = this.mathToCanvas(w11 * i + w12 * (range) + b1, w21 * i + w22 * (range) + b2);

      ctx.beginPath();
      ctx.moveTo(p1.px, p1.py);
      ctx.lineTo(p2.px, p2.py);
      ctx.stroke();

      // Lines where y = i (varying x from -range to range)
      const q1 = this.mathToCanvas(w11 * (-range) + w12 * i + b1, w21 * (-range) + w22 * i + b2);
      const q2 = this.mathToCanvas(w11 * (range) + w12 * i + b1, w21 * (range) + w22 * i + b2);

      ctx.beginPath();
      ctx.moveTo(q1.px, q1.py);
      ctx.lineTo(q2.px, q2.py);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Draw Vector with arrow head
  drawVector(fromX, fromY, toX, toY, color = '#0284c7', label = '', lineWidth = 3) {
    const ctx = this.ctx;
    const p1 = this.mathToCanvas(fromX, fromY);
    const p2 = this.mathToCanvas(toX, toY);

    const dx = p2.px - p1.px;
    const dy = p2.py - p1.py;
    const length = Math.hypot(dx, dy);

    if (length < 2) return;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';

    // Shaft
    ctx.beginPath();
    ctx.moveTo(p1.px, p1.py);
    ctx.lineTo(p2.px, p2.py);
    ctx.stroke();

    // Arrowhead
    const headLength = Math.min(14, Math.max(8, length * 0.2));
    const angle = Math.atan2(dy, dx);
    const arrowAngle = Math.PI / 7;

    ctx.beginPath();
    ctx.moveTo(p2.px, p2.py);
    ctx.lineTo(
      p2.px - headLength * Math.cos(angle - arrowAngle),
      p2.py - headLength * Math.sin(angle - arrowAngle)
    );
    ctx.lineTo(
      p2.px - headLength * Math.cos(angle + arrowAngle),
      p2.py - headLength * Math.sin(angle + arrowAngle)
    );
    ctx.closePath();
    ctx.fill();

    // Label with outline for high contrast
    if (label) {
      const isDark = this.mode === 'dark';
      const haloColor = isDark ? (this.theme === 'retro-studio' ? '#24151f' : '#18181b') : '#ffffff';
      ctx.font = 'bold 12px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.strokeStyle = haloColor;
      ctx.lineWidth = 3.5;
      ctx.strokeText(` ${label}`, p2.px + 4, p2.py - 4);
      ctx.fillStyle = color;
      ctx.fillText(` ${label}`, p2.px + 4, p2.py - 4);
    }
    ctx.restore();
  }

  // Draw Polygon (e.g. Unit Square)
  drawPolygon(points, fillColor = 'rgba(56, 189, 248, 0.22)', strokeColor = '#0284c7', lineWidth = 2.5, dashed = false) {
    if (!points || points.length < 2) return;
    const ctx = this.ctx;
    ctx.save();

    if (dashed) ctx.setLineDash([5, 5]);
    ctx.fillStyle = fillColor;
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;

    ctx.beginPath();
    const first = this.mathToCanvas(points[0][0], points[0][1]);
    ctx.moveTo(first.px, first.py);

    for (let i = 1; i < points.length; i++) {
      const p = this.mathToCanvas(points[i][0], points[i][1]);
      ctx.lineTo(p.px, p.py);
    }
    ctx.closePath();
    if (fillColor) ctx.fill();
    if (strokeColor) ctx.stroke();
    ctx.restore();
  }

  // Draw Rotated Ellipse for PCA covariance
  drawEllipse(cx, cy, rx, ry, angleRad, strokeColor = '#0284c7', fillColor = 'rgba(186, 230, 253, 0.28)') {
    const ctx = this.ctx;
    const center = this.mathToCanvas(cx, cy);
    const px_rx = rx * this.scale;
    const px_ry = ry * this.scale;

    if (px_rx <= 0 || px_ry <= 0) return;

    ctx.save();
    ctx.translate(center.px, center.py);
    // Canvas y is flipped compared to math y, so angle is -angleRad
    ctx.rotate(-angleRad);

    ctx.beginPath();
    ctx.ellipse(0, 0, px_rx, px_ry, 0, 0, 2 * Math.PI);
    if (fillColor) {
      ctx.fillStyle = fillColor;
      ctx.fill();
    }
    if (strokeColor) {
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.stroke();
    }
    ctx.restore();
  }

  // Draw Circle in Cartesian space
  drawCircle(cx, cy, radius, fillColor = null, strokeColor = '#000000', lineWidth = 1.5, dashed = false) {
    const ctx = this.ctx;
    const center = this.mathToCanvas(cx, cy);
    const px_r = radius * this.scale;
    if (px_r <= 0) return;

    ctx.save();
    if (dashed) ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(center.px, center.py, px_r, 0, 2 * Math.PI);
    if (fillColor) {
      ctx.fillStyle = fillColor;
      ctx.fill();
    }
    if (strokeColor) {
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    }
    ctx.restore();
  }

  // Draw Angle Arc between two angles (in radians), always drawing the minor arc (<= 180 deg)
  drawAngleArc(cx, cy, radius, startAngleRad, endAngleRad, strokeColor = '#ea580c', lineWidth = 2) {
    const ctx = this.ctx;
    const center = this.mathToCanvas(cx, cy);
    const px_r = radius * this.scale;
    if (px_r <= 0) return;

    // Normalize angular difference in math space to [-PI, PI]
    let diff = (endAngleRad - startAngleRad) % (2 * Math.PI);
    if (diff > Math.PI) diff -= 2 * Math.PI;
    if (diff < -Math.PI) diff += 2 * Math.PI;

    if (Math.abs(diff) < 0.01) return; // Angles are virtually identical

    // In canvas, y is inverted, so canvas angle = -mathAngle
    // When math angle increases (diff > 0), canvas angle decreases (anticlockwise = true)
    // When math angle decreases (diff < 0), canvas angle increases (anticlockwise = false)
    const aStart = -startAngleRad;
    const aEnd = -(startAngleRad + diff);
    const anticlockwise = diff > 0;

    ctx.save();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.arc(center.px, center.py, px_r, aStart, aEnd, anticlockwise);
    ctx.stroke();
    ctx.restore();
  }

  // Render draggable handles with theme-aware styling
  drawHandles() {
    const ctx = this.ctx;
    const isRetro = this.theme === 'retro-studio';
    const isDark = this.mode === 'dark';

    let borderStroke = '#000000';
    let shadowFill = '#000000';
    let labelColor = '#000000';

    if (isRetro) {
      borderStroke = isDark ? '#fb7185' : '#4a2835';
      shadowFill = isDark ? '#12070e' : '#4a2835';
      labelColor = isDark ? '#12070e' : '#4a2835';
    } else {
      borderStroke = isDark ? '#71717a' : '#000000';
      shadowFill = isDark ? '#000000' : '#000000';
      labelColor = isDark ? '#18181b' : '#000000';
    }

    for (const h of this.handles) {
      const { px, py } = this.mathToCanvas(h.x, h.y);
      ctx.save();

      // Outer shadow
      ctx.beginPath();
      ctx.arc(px + 1.5, py + 1.5, h.radius, 0, 2 * Math.PI);
      ctx.fillStyle = shadowFill;
      ctx.fill();

      // Main handle circle
      ctx.beginPath();
      ctx.arc(px, py, h.radius, 0, 2 * Math.PI);
      ctx.fillStyle = h.color || '#fed7aa';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = borderStroke;
      ctx.stroke();

      if (h.label) {
        ctx.fillStyle = labelColor;
        ctx.font = 'bold 11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(h.label, px, py + 0.5);
      }
      ctx.restore();
    }
  }
}
