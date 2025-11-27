# Frontend - Reddit Year Prediction Website

Next.js frontend for the Reddit Year Prediction project. A modern, responsive web interface for demonstrating the ML model.

## Overview

The frontend is a Next.js 14 application that provides an interactive demo for the Reddit year prediction model. It features a polished UI with sections for project overview, methodology, interactive demo, and visualizations.

## Structure

```
frontend/
├── app/                    # Next.js app directory
│   ├── api/               # API routes (proxies to backend)
│   │   └── predict/       # Prediction endpoint proxy
│   ├── layout.tsx         # Root layout
│   ├── page.tsx           # Home page
│   └── globals.css        # Global styles
├── components/            # React components
│   ├── ui/                # Reusable UI components
│   │   ├── button.tsx
│   │   ├── card.tsx
│   │   └── textarea.tsx
│   ├── demo.tsx           # Interactive prediction demo
│   ├── hero.tsx           # Hero section
│   ├── navbar.tsx         # Navigation bar
│   ├── methodology.tsx    # Methodology section
│   └── ...                # Other feature components
├── lib/                   # Utilities
│   ├── config.ts          # Configuration (GitHub links, etc.)
│   └── utils.ts           # Helper functions
├── public/                # Static assets
│   ├── *.png              # Visualization images
│   ├── favicon.png        # Site favicon
│   └── mcgill-logo.png    # McGill logo
└── package.json           # Dependencies
```

## Setup

### 1. Install Dependencies

```bash
npm install
# or
yarn install
# or
pnpm install
```

### 2. Configuration

Create a `.env.local` file (optional):

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

If not set, the frontend will default to `http://localhost:8000` for the backend API.

### 3. Start Development Server

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

The app will be available at **http://localhost:3000**

## Building for Production

```bash
npm run build
npm start
```

Or use a process manager like PM2:

```bash
npm run build
pm2 start npm --name "reddit-frontend" -- start
```

## Features

### Interactive Demo
- Real-time prediction interface
- Paste any Reddit-style comment
- Displays predicted time period (2008-2010 or 2020-2022)
- Clean, modern UI with visual feedback

### Sections
- **Hero**: Project introduction and key stats
- **Overview**: Project description and goals
- **Team**: Team member information
- **What It Does**: Feature explanation
- **Dataset**: Data source and preprocessing
- **Methodology**: Technical approach and pipeline
- **Demo**: Interactive prediction interface
- **Conclusion**: Summary and future work

### UI Features
- Responsive design (mobile, tablet, desktop)
- Dark mode support (toggle in navbar)
- Smooth animations and transitions
- Accessible components (ARIA labels, keyboard navigation)

## API Integration

The frontend communicates with the backend through:

1. **Next.js API Route** (`app/api/predict/route.ts`): Proxies requests to the FastAPI backend
2. **Direct Backend Calls**: The proxy handles CORS and routing

### Making Predictions

The demo component sends requests to `/api/predict`:

```typescript
const response = await fetch('/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: inputText.trim(),
    task_type: 'classification',
  }),
})
```

## Static Assets

Place visualization images in `public/`:
- `confusion-matrix.png`
- `prediction-vs-actual.png`
- `error-distribution.png`
- `temporal-accuracy.png`
- `word-cloud-*.png` (optional)

The visualizations component will display placeholders if images are missing.

## Development

### Adding New Components

1. Create component file in `components/`
2. Import and use in `app/page.tsx`
3. Add navigation link in `components/navbar.tsx` if needed

### Styling

- **Tailwind CSS**: Utility-first CSS framework
- **shadcn/ui**: Component library built on Radix UI
- **Custom CSS**: Global styles in `app/globals.css`

### TypeScript

The project uses TypeScript for type safety. Component props should be typed:

```typescript
interface ComponentProps {
  title: string
  description?: string
}
```

## Troubleshooting

**API connection errors?**
- Ensure the backend is running on `http://localhost:8000`
- Check `NEXT_PUBLIC_API_URL` in `.env.local`
- Verify the backend health endpoint: `curl http://localhost:8000/health`

**Images not loading?**
- Ensure image files are in `public/` directory
- Check file names match exactly (case-sensitive)
- Verify file permissions

**Build errors?**
- Clear `.next` directory: `rm -rf .next`
- Reinstall dependencies: `rm -rf node_modules && npm install`
- Check Node.js version (requires Node 18+)

## Dependencies

Key dependencies:
- `next` - React framework
- `react` - UI library
- `tailwindcss` - CSS framework
- `lucide-react` - Icon library
- `@radix-ui/*` - UI primitives (via shadcn/ui)

See `package.json` for the complete list.

## Deployment

### Vercel (Recommended)

1. Push code to GitHub
2. Import project in Vercel
3. Set environment variables
4. Deploy

### Other Platforms

The app can be deployed to any platform that supports Next.js:
- Netlify
- AWS Amplify
- Railway
- Docker (see Dockerfile if available)

## Environment Variables

- `NEXT_PUBLIC_API_URL` - Backend API URL (default: `http://localhost:8000`)
- `NEXT_PUBLIC_GITHUB_REPO` - GitHub repository URL (optional)
