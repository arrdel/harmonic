# HARMONIC Assets Directory

This directory contains visual assets for the HARMONIC proposal documentation.

## Required Images

### 1. `harmonic_banner.png`
**Description:** A wide banner image showing the HARMONIC concept
- **Dimensions:** 1920 x 400 px
- **Content:** 
  - Left side: Text prompt bubble + Image prompt thumbnail with conflict symbol (⚡) between them
  - Center: HARMONIC logo/wordmark with flowing lines connecting inputs to output
  - Right side: Beautiful harmonized artwork output
- **Color scheme:** Deep purple/blue gradient with gold accents
- **Style:** Modern, clean, technical but artistic

### 2. `teaser_comparison.png`
**Description:** Side-by-side comparison of MGAD vs HARMONIC results
- **Dimensions:** 1600 x 600 px
- **Layout:**
  ```
  ┌─────────────────────────────────────────────────────────────────┐
  │  PROMPTS (top center)                                           │
  │  Text: "Serene mountain landscape, impressionist style"         │
  │  Image: [cubist cityscape thumbnail]                            │
  ├─────────────────────────────┬───────────────────────────────────┤
  │       MGAD Output           │       HARMONIC Output             │
  │   (artifacts/incoherent)    │   (harmonized, coherent)          │
  │                             │                                   │
  │   [generated image]         │   [generated image]               │
  │                             │                                   │
  │   ✗ Failed to reconcile     │   ✓ Combines mountain forms       │
  │                             │     with impressionist brushwork  │
  └─────────────────────────────┴───────────────────────────────────┘
  ```

### 3. `harmonic_architecture.png`
**Description:** Detailed architecture diagram
- **Dimensions:** 1400 x 1000 px
- **Content:** Full system diagram showing:
  - Input stage (text/image prompts → CLIP encoders)
  - HARMONIC module (conflict detector → fusion → scheduler)
  - Output stage (harmonized guidance → diffusion model)
- **Style:** Technical diagram with boxes, arrows, and annotations
- **Colors:** 
  - Blue for text pathway
  - Orange for image pathway
  - Purple for fused/output pathway
  - Gray for diffusion model components

### 4. `failure_cases.png`
**Description:** Grid of MGAD failure cases with conflicting prompts
- **Dimensions:** 1200 x 800 px
- **Layout:** 2x3 grid showing:
  - Row 1: Input prompts (text + image thumbnail)
  - Row 2: Failed MGAD outputs with artifacts
- **Annotations:** Red circles highlighting artifacts, brief failure descriptions

### 5. `conflict_spectrum.png` (optional)
**Description:** Visual representation of prompt alignment spectrum
- **Dimensions:** 1000 x 300 px
- **Content:** Horizontal spectrum from "Aligned" to "Conflicting" with example pairs

### 6. `temporal_scheduling.png` (optional)
**Description:** Graph showing guidance weight evolution over diffusion timesteps
- **Dimensions:** 800 x 500 px
- **Content:** 
  - X-axis: Timestep (T → 0)
  - Y-axis: Weight (0 → 1)
  - Three lines: w_text, w_img, conflict modulation
  - Annotations for "early stage", "mid stage", "late stage"

---

## Image Generation Guidelines

These images can be created using:
1. **Diagrams:** Figma, draw.io, or LaTeX/TikZ
2. **Generated artwork:** Run MGAD with various prompts to capture failure cases
3. **Banner/branding:** Canva or Adobe Illustrator

## Placeholder Usage

Until images are created, the README will show broken image links. 
Replace placeholders with actual images as they are created.

---

## Color Palette

| Element | Hex Code | Usage |
|---------|----------|-------|
| Primary Purple | #6B46C1 | Headers, accents |
| Secondary Blue | #3182CE | Text pathway |
| Accent Orange | #DD6B20 | Image pathway |
| Success Green | #38A169 | Checkmarks, success |
| Error Red | #E53E3E | X marks, failures |
| Background | #1A202C | Dark backgrounds |
| Text | #E2E8F0 | Light text on dark |
