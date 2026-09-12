import type { Metadata } from "next";
import React from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "EndoPredict AI | Clinical Decision Support System",
  description: "Production clinical machine learning system for chronic cardiometabolic disease early detection, conformal prediction, and counterfactual recourse.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
