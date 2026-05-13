/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./src/**/*.{js,jsx,ts,tsx}", "./index.html"],
  theme: {
    extend: {
      colors: {
        border:     "hsl(var(--border))",
        input:      "hsl(var(--input))",
        ring:       "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary:    { DEFAULT: "hsl(var(--primary))", foreground: "hsl(var(--primary-foreground))" },
        secondary:  { DEFAULT: "hsl(var(--secondary))", foreground: "hsl(var(--secondary-foreground))" },
        destructive:{ DEFAULT: "hsl(var(--destructive))", foreground: "hsl(var(--destructive-foreground))" },
        muted:      { DEFAULT: "hsl(var(--muted))", foreground: "hsl(var(--muted-foreground))" },
        accent:     { DEFAULT: "hsl(var(--accent))", foreground: "hsl(var(--accent-foreground))" },
        card:       { DEFAULT: "hsl(var(--card))", foreground: "hsl(var(--card-foreground))" },
        popover:    { DEFAULT: "hsl(var(--popover))", foreground: "hsl(var(--popover-foreground))" },
        sidebar:    { DEFAULT: "hsl(var(--sidebar))", border: "hsl(var(--sidebar-border))" },
        indigo: {
          400: "#818cf8", 500: "#6366f1", 600: "#4f46e5",
        },
        cyan: {
          400: "#22d3ee", 500: "#06b6d4",
        },
        emerald: {
          400: "#34d399", 500: "#10b981", 600: "#059669",
        },
        amber: {
          400: "#fbbf24", 500: "#f59e0b",
        },
        violet: {
          400: "#a78bfa", 500: "#8b5cf6",
        },
        zinc: {
          400: "#a1a1aa", 500: "#71717a", 600: "#52525b",
          700: "#3f3f46", 800: "#27272a", 900: "#18181b", 950: "#09090b",
        },
      },
      borderRadius: {
        lg: "var(--radius)", md: "calc(var(--radius) - 2px)", sm: "calc(var(--radius) - 4px)",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
      },
      keyframes: {
        "fade-in": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        "fade-in-scale": {
          from: { opacity: "0", transform: "scale(.97)" },
          to:   { opacity: "1", transform: "scale(1)" },
        },
        "slide-in-right": {
          from: { opacity: "0", transform: "translateX(10px)" },
          to:   { opacity: "1", transform: "translateX(0)" },
        },
        shimmer: {
          "0%":   { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition:  "200% 0" },
        },
        "bounce-dot": {
          "0%,80%,100%": { transform: "translateY(0)" },
          "40%":          { transform: "translateY(-4px)" },
        },
      },
      animation: {
        "fade-in":        "fade-in .2s ease both",
        "fade-in-scale":  "fade-in-scale .15s ease both",
        "slide-in-right": "slide-in-right .2s ease both",
        shimmer:          "shimmer 1.5s linear infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
