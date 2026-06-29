/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'on-surface': '#0b1c30',
        'on-surface-variant': '#434655',
        'inverse-surface': '#213145',
        'secondary-fixed': '#dae2fd',
        'outline-variant': '#c3c6d7',
        'primary-container': '#2563eb',
        'tertiary-container': '#6b6e70',
        primary: '#004ac6',
        'on-primary-fixed': '#00174b',
        'on-tertiary': '#ffffff',
        'inverse-primary': '#b4c5ff',
        'secondary-container': '#dae2fd',
        secondary: '#565e74',
        'surface-tint': '#0053db',
        'on-error': '#ffffff',
        'on-secondary-fixed-variant': '#3f465c',
        'on-secondary-fixed': '#131b2e',
        'on-tertiary-container': '#eff1f3',
        'surface-variant': '#d3e4fe',
        tertiary: '#525657',
        surface: '#f8f9ff',
        'surface-bright': '#f8f9ff',
        error: '#ba1a1a',
        'on-primary-container': '#eeefff',
        'on-background': '#0b1c30',
        'surface-container-highest': '#d3e4fe',
        'on-primary': '#ffffff',
        'on-primary-fixed-variant': '#003ea8',
        background: '#f8f9ff',
        outline: '#737686',
        'on-tertiary-fixed': '#191c1e',
        'surface-dim': '#cbdbf5',
        'on-secondary-container': '#5c647a',
        'primary-fixed': '#dbe1ff',
        'on-secondary': '#ffffff',
        'surface-container-low': '#eff4ff',
        'secondary-fixed-dim': '#bec6e0',
        'error-container': '#ffdad6',
        'surface-container-high': '#dce9ff',
        'surface-container-lowest': '#ffffff',
        'surface-container': '#e5eeff',
        'on-tertiary-fixed-variant': '#444749',
        'tertiary-fixed-dim': '#c4c7c9',
        'on-error-container': '#93000a',
        'inverse-on-surface': '#eaf1ff',
        'tertiary-fixed': '#e0e3e5',
        'primary-fixed-dim': '#b4c5ff',
        success: '#10b981'
      },
      spacing: {
        'stack-sm': '8px',
        'stack-md': '16px',
        'stack-lg': '32px',
        'stack-xl': '64px',
        gutter: '24px',
        'margin-x': '32px',
        'container-max': '1440px'
      },
      maxWidth: {
        'container-max': '1440px'
      },
      fontFamily: {
        'headline-lg': ['"Hanken Grotesk"', 'sans-serif'],
        'headline-lg-mobile': ['"Hanken Grotesk"', 'sans-serif'],
        'display-lg': ['"Hanken Grotesk"', 'sans-serif'],
        'title-md': ['Inter', 'sans-serif'],
        'body-md': ['Inter', 'sans-serif'],
        'body-sm': ['Inter', 'sans-serif'],
        'metric-value': ['"JetBrains Mono"', 'monospace'],
        'metric-label': ['"JetBrains Mono"', 'monospace']
      },
      fontSize: {
        'display-lg': ['48px', { lineHeight: '56px', letterSpacing: '-0.02em', fontWeight: '700' }],
        'headline-lg': ['32px', { lineHeight: '40px', letterSpacing: '-0.01em', fontWeight: '600' }],
        'headline-lg-mobile': ['24px', { lineHeight: '32px', fontWeight: '600' }],
        'title-md': ['18px', { lineHeight: '28px', fontWeight: '600' }],
        'body-md': ['16px', { lineHeight: '24px', fontWeight: '400' }],
        'body-sm': ['14px', { lineHeight: '20px', fontWeight: '400' }],
        'metric-value': ['14px', { lineHeight: '20px', fontWeight: '600' }],
        'metric-label': ['12px', { lineHeight: '16px', letterSpacing: '0.05em', fontWeight: '500' }]
      }
    }
  },
  plugins: []
};
