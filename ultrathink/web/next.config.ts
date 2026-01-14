import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable static export for serving from FastAPI
  output: "export",

  // No base path needed when served from root
  basePath: "",

  // Use trailing slashes for static file compatibility
  trailingSlash: true,

  // Disable image optimization for static export
  images: {
    unoptimized: true,
  },

  // Disable ESLint during build (we run it separately)
  eslint: {
    ignoreDuringBuilds: true,
  },

  // Disable TypeScript errors during build (we run it separately)
  typescript: {
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
