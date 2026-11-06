---
name: "SocraticCoach Studio System"
version: "2.1.0-extended"
schema: "design.md/v2"

colors:
  canvas:
    background: "oklch(14% 0.012 260)"
    subtle: "oklch(16% 0.014 260)"
    surface: "oklch(19% 0.016 260)"
    surface-elevated: "oklch(22% 0.018 260)"
    surface-hover: "oklch(25% 0.022 260)"
    surface-active: "oklch(21% 0.020 260)"
  borders:
    subtle: "oklch(24% 0.015 260)"
    default: "oklch(30% 0.020 260)"
    strong: "oklch(40% 0.025 260)"
    focus-ring: "oklch(75% 0.140 250)"
  typography:
    primary: "oklch(96% 0.005 260)"
    secondary: "oklch(78% 0.012 260)"
    muted: "oklch(58% 0.015 260)"
    inverse: "oklch(12% 0.010 260)"
  brand:
    primary: "oklch(96% 0.005 260)"
    primary-hover: "oklch(88% 0.008 260)"
    primary-active: "oklch(80% 0.010 260)"
    on-primary: "oklch(12% 0.010 260)"
    accent: "oklch(72% 0.160 250)"
    accent-subtle: "oklch(25% 0.060 250)"
  status:
    idle: "oklch(62% 0.020 260)"
    listening: "oklch(75% 0.180 145)"
    thinking: "oklch(82% 0.160 85)"
    speaking: "oklch(70% 0.190 310)"
    muted: "oklch(65% 0.220 25)"
    warning: "oklch(80% 0.170 70)"

typography:
  fontFamilies:
    sans: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
    mono: "'JetBrains Mono', 'Fira Code', ui-monospace, SFMono-Regular, monospace"
  scale:
    display-xl:
      fontFamily: "{typography.fontFamilies.sans}"
      fontSize: "2.75rem"
      lineHeight: "1.1"
      fontWeight: "700"
      letterSpacing: "-0.035em"
    headline-lg:
      fontFamily: "{typography.fontFamilies.sans}"
      fontSize: "1.75rem"
      lineHeight: "1.2"
      fontWeight: "600"
      letterSpacing: "-0.025em"
    title-md:
      fontFamily: "{typography.fontFamilies.sans}"
      fontSize: "1.25rem"
      lineHeight: "1.3"
      fontWeight: "600"
      letterSpacing: "-0.015em"
    body-lg:
      fontFamily: "{typography.fontFamilies.sans}"
      fontSize: "1.125rem"
      lineHeight: "1.5"
      fontWeight: "400"
      letterSpacing: "0.0em"
    body-md:
      fontFamily: "{typography.fontFamilies.sans}"
      fontSize: "0.9375rem"
      lineHeight: "1.45"
      fontWeight: "400"
      letterSpacing: "0.0em"
    body-sm:
      fontFamily: "{typography.fontFamilies.sans}"
      fontSize: "0.8125rem"
      lineHeight: "1.4"
      fontWeight: "400"
      letterSpacing: "0.005em"
    mono-metric:
      fontFamily: "{typography.fontFamilies.mono}"
      fontSize: "0.875rem"
      lineHeight: "1.2"
      fontWeight: "500"
      letterSpacing: "-0.01em"
      fontFeatureSettings: "'tnum' on, 'zero' on"
    mono-code:
      fontFamily: "{typography.fontFamilies.mono}"
      fontSize: "0.8125rem"
      lineHeight: "1.5"
      fontWeight: "400"

spacing:
  baseline: "4px"
  xxs: "4px"
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "24px"
  xl: "32px"
  xxl: "48px"

radii:
  none: "0px"
  xs: "4px"
  sm: "6px"
  md: "8px"
  lg: "12px"
  xl: "16px"
  pill: "9999px"

elevation:
  none: "none"
  card-resting: "0 0 0 1px {colors.borders.subtle}"
  card-elevated: "0 0 0 1px {colors.borders.default}, 0 4px 12px -2px oklch(0% 0 0 / 45%)"
  card-active-glow: "0 0 0 1px {colors.borders.strong}, 0 0 20px -4px {colors.brand.accent-subtle}"
  alert-question: "0 0 0 1px {colors.status.thinking}, 0 8px 32px -4px oklch(0% 0 0 / 65%)"

motion:
  springs:
    snappy:
      stiffness: 340
      damping: 28
      mass: 0.8
      velocity: 0
    smooth:
      stiffness: 190
      damping: 22
      mass: 1.0
      velocity: 0
    gentle:
      stiffness: 120
      damping: 18
      mass: 1.2
      velocity: 0
  gestures:
    pressScale: 0.98
    dragThreshold: 6
  durations:
    instant: "80ms"
    fast: "150ms"
    normal: "240ms"
  reducedMotion:
    transform: "none"
    transitionFallback: "opacity 120ms ease-out"

components:
  questionCard:
    padding: "{spacing.lg}"
    borderRadius: "{radii.lg}"
    background: "{colors.canvas.surface-elevated}"
    states:
      default:
        opacity: 1
        transform: "none"
        border: "1px solid {colors.borders.default}"
        elevation: "{elevation.card-resting}"
      hover:
        border: "1px solid {colors.borders.strong}"
        elevation: "{elevation.card-elevated}"
      active:
        transform: "scale({motion.gestures.pressScale})"
        spring: "{motion.springs.snappy}"
      focus:
        border: "1px solid {colors.borders.focus-ring}"
        ring: "2px solid {colors.brand.accent}"
        ringOffset: "2px"
      disabled:
        opacity: 0.35
        pointerEvents: "none"
      hidden:
        opacity: 0
        transform: "scale(0.96) translateY(12px)"
        pointerEvents: "none"
      incoming:
        opacity: 1
        transform: "scale(1) translateY(0)"
        spring: "{motion.springs.smooth}"
        border: "1px solid {colors.status.thinking}"
        elevation: "{elevation.alert-question}"
      listening:
        border: "1px solid {colors.status.listening}"
      resolved:
        opacity: 0
        transform: "scale(0.98) translateY(-8px)"
        duration: "{motion.durations.fast}"

  statusPill:
    height: "28px"
    padding: "0 {spacing.sm}"
    borderRadius: "{radii.pill}"
    typography: "{typography.scale.mono-metric}"
    states:
      default:
        background: "{colors.canvas.surface}"
        border: "1px solid {colors.borders.subtle}"
        textColor: "{colors.status.idle}"
        dotColor: "{colors.status.idle}"
      hover:
        background: "{colors.canvas.surface-hover}"
        border: "1px solid {colors.borders.default}"
      active:
        background: "{colors.canvas.surface-active}"
        transform: "scale({motion.gestures.pressScale})"
      focus:
        ring: "2px solid {colors.brand.accent}"
        ringOffset: "2px"
      disabled:
        opacity: 0.35
        cursor: "not-allowed"
      idle:
        textColor: "{colors.status.idle}"
        dotColor: "{colors.status.idle}"
      listening:
        border: "1px solid {colors.status.listening}"
        textColor: "{colors.typography.primary}"
        dotColor: "{colors.status.listening}"
        pulse: true
      thinking:
        border: "1px solid {colors.status.thinking}"
        textColor: "{colors.status.thinking}"
        dotColor: "{colors.status.thinking}"
        pulse: true
      speaking:
        border: "1px solid {colors.status.speaking}"
        textColor: "{colors.status.speaking}"
        dotColor: "{colors.status.speaking}"
        pulse: false
      muted:
        border: "1px solid {colors.status.muted}"
        textColor: "{colors.status.muted}"
        dotColor: "{colors.status.muted}"
        pulse: false

  actionButton:
    height: "36px"
    padding: "0 {spacing.md}"
    borderRadius: "{radii.md}"
    typography: "{typography.scale.body-sm}"
    states:
      default:
        background: "{colors.canvas.surface}"
        border: "1px solid {colors.borders.default}"
        textColor: "{colors.typography.primary}"
        transform: "scale(1)"
      hover:
        background: "{colors.canvas.surface-hover}"
        border: "1px solid {colors.borders.strong}"
        spring: "{motion.springs.snappy}"
      active:
        background: "{colors.canvas.surface-active}"
        transform: "scale({motion.gestures.pressScale})"
      focus:
        border: "1px solid {colors.borders.focus-ring}"
        ring: "2px solid {colors.brand.accent}"
        ringOffset: "2px"
      disabled:
        opacity: 0.35
        cursor: "not-allowed"
        transform: "none"

  audioWaveform:
    height: "48px"
    background: "{colors.canvas.subtle}"
    borderRadius: "{radii.md}"
    border: "1px solid {colors.borders.subtle}"
    barWidth: "3px"
    barGap: "2px"
    states:
      default:
        barColor: "{colors.status.idle}"
        opacity: 0.3
      hover:
        border: "1px solid {colors.borders.default}"
      active:
        barColor: "{colors.status.listening}"
        opacity: 1
      focus:
        ring: "2px solid {colors.brand.accent}"
      disabled:
        opacity: 0.35
      idle:
        barColor: "{colors.status.idle}"
        opacity: 0.3
      listening:
        barColor: "{colors.status.listening}"
        opacity: 1
      muted:
        barColor: "{colors.status.muted}"
        opacity: 0.2

  transcriptFeed:
    background: "{colors.canvas.surface}"
    border: "1px solid {colors.borders.subtle}"
    borderRadius: "{radii.lg}"
    padding: "{spacing.md}"
    typography: "{typography.scale.body-lg}"
    states:
      default:
        textColor: "{colors.typography.primary}"
      hover:
        border: "1px solid {colors.borders.default}"
      active:
        background: "{colors.canvas.surface-active}"
      focus:
        ring: "2px solid {colors.brand.accent}"
      disabled:
        opacity: 0.35
      latestSentence:
        textColor: "{colors.typography.primary}"
        fontWeight: "600"
      historicalSentence:
        textColor: "{colors.typography.muted}"
        fontWeight: "400"

---

# SocraticCoach Studio System

## 1. Overview & System Purpose

SocraticCoach is a local, offline teaching assistant operated during a live lecture. The HUD is designed for an instructor who is speaking, moving, and monitoring a room rather than sitting at a desk. That context rules out visual clutter, multi-level dropdowns, hidden state, and low-contrast text: each competes with lecture delivery and makes a time-critical action harder to find.

The HUD has two complementary modes of attention. Peripheral glanceability provides high-visibility status for microphone, inference, speech playback, clock, and hardware health without demanding focus. Focal clarity takes over when a student question appears: the question card becomes the singular visual priority, exposes latency and the next action, and remains readable at a standing distance of 3–6 feet.

The visual language deliberately converges modern studio darkness and technical austerity, as seen in Hyperbrowser and Linear, with Apple developer-tool typographic rhythm and shadcn-style modularity. This restraint is paired with accessible, predictable finite state machines. A state must be visible, named, and keyboard-reachable; appearance must never be the only signal.

## 2. Colors & WCAG AAA Contrast Mandates

All surfaces use the OKLCH dark-slate foundation in the `colors.canvas` tokens. Pure `#000000` is forbidden for card containers because it produces an unnecessarily harsh luminance boundary, collapses shadow and border information, and performs poorly under classroom projector light. Subtle blue/slate chroma preserves depth and reduces eye fatigue while keeping the interface dark.

The one-pixel border elevation model replaces blurred shadow stacks in this high-density technical interface. `borders.subtle` defines adjacent structure, `borders.default` defines a normal interactive boundary, and `borders.strong` marks elevated or hovered content. Shadows are secondary and never the sole carrier of hierarchy.

Body typography on surface containers must exceed a 7:1 contrast ratio (WCAG AAA). Implementations must verify the rendered foreground/background pair, not merely compare token names. Primary transcript text uses `typography.primary`; secondary labels use `typography.secondary`; muted text is reserved for historical or nonessential metadata and must not carry required instructions.

Semantic colors are exclusive. Green (`status.listening`) means acoustic speech capture. Amber (`status.thinking`) means local LLM inference. Violet (`status.speaking`) means student TTS verbal playback. Red (`status.muted`) means hardware or software microphone muting. Do not reuse these hues for decorative accents or unrelated success, loading, or error states.

## 3. Typography & Distance Scannability

Inter is used for human speech prose, transcript content, question text, labels, and controls because its open forms remain legible at distance. JetBrains Mono is used for system metrics, timestamps, latency, counters, and telemetry because its fixed character widths support rapid comparison.

All clocks, latency values, VRAM counters, CPU loads, and other frequently changing metrics must set `font-variant-numeric: tabular-nums` and `font-feature-settings: "tnum" on, "zero" on`. This prevents width changes from causing layout jitter during high-frequency updates.

At 3–6 feet, live transcript and question prose must use at least the `body-lg` size (1.125rem) with a 1.5 line height. The question heading uses `headline-lg` or larger. Auxiliary hardware metrics may use `mono-metric` (0.875rem) only when paired with a clearly visible icon, label, or color state; critical mute status must never be communicated by small text alone. No required action may depend on text smaller than `body-sm`.

## 4. Layout Architecture & Spatial Rhythm

The spatial baseline is 4px, with an 8px rhythm for primary gaps. Every margin, padding, border-adjacent gap, and fixed height must be a multiple of 4px; main panel spacing should use 8px or 16px multiples. Avoid arbitrary fractional spacing and avoid introducing a second spacing scale.

The cockpit uses a 12-column responsive grid with a minimum 24px outer gutter:

* **Top navigation/header bar:** one 48px row containing the clock, session timer, hardware telemetry pills (VRAM and CPU STT load), and global microphone state.
* **Main viewport:** a 60/40 or 70/30 split. The primary left panel is a live, auto-scrolling lecture transcript optimized for reading while speaking. The center-top stage is the Socratic Question Alert Box, which pops in only when a question is triggered and contains latency, question text, and single-keystroke shortcuts. The secondary right panel is collapsible and contains Blackboard Vision OCR and diagram feed.
* **Bottom anchor bar:** a persistent real-time audio energy waveform with the ambient noise-floor calibration line.

At narrower laptop widths, collapse the right panel before reducing transcript type size. At projector and 1440p widths, preserve the transcript reading measure and use the additional columns for breathing room rather than extra controls.

## 5. Elevation, Borders & Glass Philosophy

Traditional glassmorphism is banned: heavy blur, translucent white highlights, and layered backdrop filters consume GPU fill-rate and create unstable legibility behind dynamic text. This is especially inappropriate while local Whisper and Ollama processes compete for compute.

Use structural one-pixel OKLCH borders to define hierarchy. A resting card uses `card-resting`; an elevated or hovered card may use `card-elevated`; an active card uses a stronger border. Keep surfaces opaque enough that transcript text never competes with content behind it.

The active Question Card uses `elevation.alert-question`: a single-color amber radial/edge glow associated with inference. It must attract peripheral vision without flashing, strobing, pulsing brightness, or using a multicolor gradient. The glow is subordinate to the card's text and state label.

## 6. Motion & Interaction (Emil Kowalski Heuristics)

**Restraint first.** The initial dashboard render has zero entrance animations. It must be immediately interactive and usable if animation is interrupted.

**Springs over Bezier curves.** Interactive micro-animations, including card appearance, button press scale, drawer opening, and state-pill transitions, use the explicit `snappy`, `smooth`, or `gentle` spring configuration in the frontmatter. Standard CSS cubic-bezier easing is prohibited for interactive states. The reduced-motion fallback is the sole exception.

**Compositor-only animations.** Never animate `height`, `width`, `top`, `left`, `margin`, or `padding`. Animate only `transform` (`scale` and `translate3d`) and `opacity`; layout measurement must occur outside the animation loop. This avoids reflow and protects the UI thread used alongside Whisper and Ollama inference.

**Reduced motion contract.** When `@media (prefers-reduced-motion: reduce)` is enabled, set all transforms and translations to `none`, disable spring overshoot and pulses, and use only the declared `opacity 120ms ease-out` cross-fade. Content order and state timing remain identical; only visual interpolation changes.

## 7. Component Interaction Specifications & Finite States

### 7.1 Socratic Question Alert Box

When the local model emits a question-generation trigger, create the question in the `incoming` state, attach its timestamp and latency, and apply the `smooth` spring. Do not steal focus from an instructor who is typing; announce the question through an assertive live region while preserving a visible focus ring for keyboard navigation. The card then enters `default`/reading state and exposes Space (acknowledge), Esc (dismiss), and the current status.

Space marks the question addressed and transitions through `active` to `resolved`; Esc dismisses immediately and records the dismissal. If the instructor resumes speech, the card may auto-dismiss only after the speech detector confirms resumed capture, never on a transient noise spike. A disabled or unavailable question remains visible with the `disabled` state and an explicit reason.

### 7.2 Hardware Telemetry & Mute Pill

The finite state sequence is `IDLE → LISTENING → THINKING → SPEAKING → MUTED`. `IDLE` uses slate; `LISTENING` changes the border and dot to green and may pulse at a restrained rate; `THINKING` changes them to amber; `SPEAKING` changes them to violet without a pulse; and `MUTED` changes them to red with a solid, high-visibility indicator. Transitions update the accessible label and never rely on color alone.

The SVG status dot may animate opacity or a compositor-only scale pulse for `LISTENING` and `THINKING`. It must stop immediately in `SPEAKING` and `MUTED`. The Mute pill remains visible from across the room and exposes the same state to screen readers.

### 7.3 Rolling Transcript Feed

Append recognized utterances in order and keep the scroll container anchored to the newest line only while the instructor is at the bottom. If the instructor scrolls upward, pause auto-scroll, show a nonblocking “new speech” affordance, and never jerk the viewport. Activating the affordance or returning to the bottom resumes smooth compositor-only scrolling.

Utterances older than 60 seconds transition to `historicalSentence` styling and dim to `typography.muted`; the text remains selectable and accessible. Never dim the currently relevant question or the newest sentence below the contrast mandate.

### 7.4 Real-Time Audio Waveform

Render waveform bars in a 60fps canvas or equivalent compositor-friendly layer using the latest audio analysis buffer. Normalize energy against the calibrated ambient noise floor, clamp low-level noise to the idle baseline, and preserve a visible floor line. The render loop must not allocate DOM nodes or trigger layout.

When the mute guard is active, set every bar to a dead-flat line and use the red muted state. Calibration (`C`) samples the ambient floor without speech, updates the line atomically, and announces completion. If the audio stream is unavailable, show the disabled state and an explicit error rather than silently presenting idle audio.

## 8. Accessible Keyboard Control & Shortcut Map

Shortcuts are active only when focus is not inside a text-entry control, unless the control explicitly opts into cockpit shortcuts. Each action has a visible label, a screen-reader name, and a focusable alternative.

| Key | Operation |
| --- | --- |
| `Space` | Mark the current question as addressed / acknowledge. |
| `M` | Toggle hardware/software microphone mute. |
| `Esc` | Dismiss the active question immediately. |
| `F` | Toggle Fullscreen Presentation / Cockpit Mode. |
| `C` | Recalibrate the ambient microphone noise floor. |

Focus indicators use `borders.focus-ring` and the declared ring offset. Keyboard actions must be idempotent where possible and must announce state changes in a live region.

## 9. Strict Do's and Don'ts for Coding Agents

### Do

* Use semantic OKLCH tokens rather than hardcoded hex values.
* Set `font-feature-settings: "tnum" on, "zero" on` and `font-variant-numeric: tabular-nums` on real-time counters and latencies.
* Enforce a minimum 7:1 contrast ratio on lecture transcripts and required body text.
* Make the mute status indicator visible from across the room and expose it textually.
* Implement every interactive component's `default`, `hover`, `active`, `focus`, `disabled`, and contextual states.
* Keep all core teaching workflows operable from the keyboard.
* Use spring tokens and transform/opacity-only animation on the UI thread.

### Don't

* Do not introduce heavy drop shadows, multi-color gradient text, or gratuitous bounce curves.
* Do not animate layout properties such as `width`, `height`, `margin`, or `padding`.
* Do not require mouse interactions for core teaching workflows.
* Do not use pure `#000000` or pure `#FFFFFF`.
* Do not reuse green, amber, violet, or red status colors for unrelated semantics.
* Do not hide errors behind an idle-looking fallback, silently discard unavailable audio, or make color the only state signal.
