"use client";

import { useState, useRef, useCallback } from "react";
import styles from "./JaundiceScan.module.css";

type PredictionResult = {
  prediction: "Jaundice" | "Normal";
  confidence: number;
  probability: number;
  risk_level: "High" | "Moderate" | "Low";
  message: string;
};

export default function JaundiceScan() {
  const [image, setImage] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const handleFile = (f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Please upload a valid image file.");
      return;
    }
    setFile(f);
    setResult(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setImage(e.target?.result as string);
    reader.readAsDataURL(f);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) handleFile(dropped);
  }, []);

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data: PredictionResult = await res.json();
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to connect to server. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setFile(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const riskColor = result
    ? result.risk_level === "High"
      ? "#e84040"
      : result.risk_level === "Moderate"
      ? "#f0a500"
      : "#2ee87a"
    : "#2ee87a";

  const architectureFlow = [
    {
      title: "Input Image",
      detail: "Scleral eye region upload",
    },
    {
      title: "YCbCr Preprocess",
      detail: "Skin segmentation + normalization",
    },
    {
      title: "ResNet50 Backbone",
      detail: "Deep feature extraction",
    },
    {
      title: "Dense Classifier",
      detail: "Sigmoid binary prediction",
    },
    {
      title: "Output Report",
      detail: "Risk level + confidence score",
    },
  ];

  return (
    <div className={styles.page}>
      {/* Background grid */}
      <div className={styles.grid} />

      {/* Header */}
      <header className={styles.header}>
        <div className={styles.logo}>
          <span className={styles.logoIcon}>⬡</span>
          <span className={styles.logoText}>JaundiScan</span>
        </div>
        <div className={styles.badge}>ResNet50 · 91% Accuracy</div>
      </header>

      <main className={styles.main}>
        {/* Hero */}
        <section className={styles.hero}>
          <p className={styles.heroLabel}>AI-Powered Medical Imaging</p>
          <h1 className={styles.heroTitle}>
            Detect Jaundice <br />
            <em>Instantly.</em>
          </h1>
          <p className={styles.heroSub}>
            Upload a scleral (eye white) image. Our deep learning model analyzes
            YCbCr skin-tone features using ResNet50 transfer learning.
          </p>
        </section>

        {/* Upload + Result layout */}
        <div className={styles.layout}>
          {/* Upload Card */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <span className={styles.cardStep}>01</span>
              <span className={styles.cardTitle}>Upload Image</span>
            </div>

            {!image ? (
              <div
                className={`${styles.dropzone} ${dragging ? styles.dragging : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className={styles.dropIcon}>
                  <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
                    <circle cx="20" cy="20" r="19" stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 3" />
                    <path d="M20 13v14M13 20l7-7 7 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
                <p className={styles.dropText}>Drop scleral image here</p>
                <p className={styles.dropSub}>or click to browse · JPG, PNG, WEBP</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className={styles.hiddenInput}
                  onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                />
              </div>
            ) : (
              <div className={styles.preview}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={image} alt="Uploaded scleral" className={styles.previewImg} />
                <div className={styles.previewOverlay}>
                  <span className={styles.fileName}>{file?.name}</span>
                </div>
              </div>
            )}

            {image && (
              <div className={styles.actions}>
                <button className={styles.btnSecondary} onClick={handleReset}>
                  ✕ Reset
                </button>
                <button
                  className={styles.btnPrimary}
                  onClick={handleAnalyze}
                  disabled={loading}
                >
                  {loading ? (
                    <span className={styles.spinner}>
                      <span className={styles.spinnerDot} />
                      Analyzing...
                    </span>
                  ) : (
                    "Analyze →"
                  )}
                </button>
              </div>
            )}

            {error && <div className={styles.errorBox}>{error}</div>}
          </div>

          {/* Result Card */}
          <div className={styles.card}>
            <div className={styles.cardHeader}>
              <span className={styles.cardStep}>02</span>
              <span className={styles.cardTitle}>Analysis Result</span>
            </div>

            {!result && !loading && (
              <div className={styles.emptyResult}>
                <div className={styles.emptyIcon}>
                  <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                    <circle cx="24" cy="24" r="10" stroke="currentColor" strokeWidth="1.5" />
                    <circle cx="24" cy="24" r="3" fill="currentColor" opacity="0.3" />
                    <path d="M24 8v4M24 36v4M8 24h4M36 24h4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                  </svg>
                </div>
                <p className={styles.emptyText}>Upload an image and click Analyze to see results</p>
              </div>
            )}

            {loading && (
              <div className={styles.loadingResult}>
                <div className={styles.loadingRing} />
                <p>Running inference...</p>
                <p className={styles.loadingSub}>YCbCr preprocessing · ResNet50 forward pass</p>
              </div>
            )}

            {result && (
              <div className={styles.result}>
                {/* Prediction badge */}
                <div
                  className={styles.predBadge}
                  style={{ borderColor: riskColor, color: riskColor }}
                >
                  <span className={styles.predDot} style={{ background: riskColor }} />
                  {result.prediction}
                </div>

                {/* Confidence bar */}
                <div className={styles.confSection}>
                  <div className={styles.confHeader}>
                    <span>Confidence</span>
                    <span className={styles.confValue}>{result.confidence}%</span>
                  </div>
                  <div className={styles.confBar}>
                    <div
                      className={styles.confFill}
                      style={{ width: `${result.confidence}%`, background: riskColor }}
                    />
                  </div>
                </div>

                {/* Stats grid */}
                <div className={styles.statsGrid}>
                  <div className={styles.stat}>
                    <span className={styles.statLabel}>Risk Level</span>
                    <span className={styles.statValue} style={{ color: riskColor }}>
                      {result.risk_level}
                    </span>
                  </div>
                  <div className={styles.stat}>
                    <span className={styles.statLabel}>Raw Probability</span>
                    <span className={styles.statValue}>{result.probability.toFixed(4)}</span>
                  </div>
                </div>

                {/* Message */}
                <div
                  className={styles.message}
                  style={{ borderLeftColor: riskColor }}
                >
                  {result.message}
                </div>

                {result.prediction === "Jaundice" && (
                  <p className={styles.disclaimer}>
                    ⚠ This is an AI screening tool. Always consult a certified medical professional for diagnosis.
                  </p>
                )}
              </div>
            )}
          </div>
        </div>

        {/* How it works */}
        <section className={styles.howSection}>
          <h2 className={styles.howTitle}>How It Works</h2>
          <div className={styles.steps}>
            {[
              { n: "1", title: "YCbCr Segmentation", desc: "Non-skin pixels are zeroed using Cb/Cr channel thresholds (Patravali et al.)" },
              { n: "2", title: "ResNet50 Extraction", desc: "ImageNet-pretrained ResNet50 extracts deep visual features from the scleral region." },
              { n: "3", title: "Two-Phase Fine-tuning", desc: "Head trained first, then top layers unfrozen for jaundice-specific feature adaptation." },
              { n: "4", title: "Binary Classification", desc: "Sigmoid output classifies as Jaundice or Normal with calibrated confidence score." },
            ].map((s) => (
              <div key={s.n} className={styles.step}>
                <span className={styles.stepNum}>{s.n}</span>
                <h3 className={styles.stepTitle}>{s.title}</h3>
                <p className={styles.stepDesc}>{s.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Model architecture flowchart */}
        <section className={styles.archSection}>
          <h2 className={styles.archTitle}>Model Architecture Flowchart</h2>
          <p className={styles.archSub}>
            End-to-end inference pipeline from uploaded eye image to jaundice screening output.
          </p>

          <div className={styles.flowChart} role="img" aria-label="Flowchart of the jaundice model architecture">
            {architectureFlow.map((node, idx) => (
              <div key={node.title} className={styles.flowItem}>
                <div className={styles.flowNode}>
                  <span className={styles.flowNodeDot} aria-hidden />
                  <h3 className={styles.flowNodeTitle}>{node.title}</h3>
                  <p className={styles.flowNodeDesc}>{node.detail}</p>
                </div>
                {idx < architectureFlow.length - 1 && (
                  <span className={styles.flowArrow} aria-hidden>
                    →
                  </span>
                )}
              </div>
            ))}
          </div>

          <div className={styles.flowLegend}>
            <div className={styles.legendCard}>
              <p className={styles.legendLabel}>Feature Stage</p>
              <p className={styles.legendValue}>ResNet50 transfers ImageNet-learned visual representations.</p>
            </div>
            <div className={styles.legendCard}>
              <p className={styles.legendLabel}>Decision Stage</p>
              <p className={styles.legendValue}>Sigmoid head estimates jaundice probability and calibrated confidence.</p>
            </div>
          </div>
        </section>
      </main>

      <footer className={styles.footer}>
        <p>Built with ResNet50 · FastAPI · Next.js · TensorFlow</p>
        <p className={styles.footerSub}>For educational and research purposes only · Not a substitute for medical advice</p>
      </footer>
    </div>
  );
}
