# Smart Streaming Instagram Captioner - React Frontend

A modern, responsive React + TypeScript frontend for the Smart Streaming Instagram Captioner application.

## 🚀 Features

- **Modern UI/UX**: Clean, professional design with smooth animations
- **Real-time Camera Integration**: Live camera feed with frame capture
- **Video Context Analysis**: Visual frame analysis with AI-powered descriptions
- **Smart Caption Generation**: Context-aware Instagram caption generation
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **TypeScript**: Full type safety and better development experience
- **Real-time Stats**: Live metrics and performance tracking

## 🛠️ Development

### Prerequisites

- Node.js 18+
- npm or yarn
- Backend server running on port 8000

### Installation

```bash
cd frontend-react
npm install
```

### Development Server

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Build for Production

```bash
npm run build
```

### Linting

```bash
npm run lint
```

## 🎨 Design System

### Color Palette
- **Primary**: Indigo (#6366f1)
- **Secondary**: Purple (#8b5cf6)
- **Accent**: Cyan (#06b6d4)
- **Success**: Emerald (#10b981)
- **Warning**: Amber (#f59e0b)
- **Error**: Red (#ef4444)

### Typography
- **Font**: Inter (Google Fonts)
- **Weights**: 300, 400, 500, 600, 700

### Components
- Modern card-based layout
- Smooth hover animations
- Responsive grid system
- Professional button styles
- Clean form controls

## 📱 Responsive Breakpoints

- **Mobile**: < 480px
- **Tablet**: 480px - 768px
- **Desktop**: > 768px

## 🔧 Architecture

### File Structure
```
src/
├── components/          # Reusable UI components
├── hooks/              # Custom React hooks
├── types/              # TypeScript type definitions
├── utils/              # Utility functions and API service
├── App.tsx             # Main application component
├── App.css             # Main styles
├── main.tsx            # Application entry point
└── index.css           # Global styles
```

### Key Hooks
- `useCamera`: Manages camera functionality and video streaming
- Custom hooks for state management and API integration

### API Integration
- Centralized API service in `utils/api.ts`
- Type-safe API calls with TypeScript
- Error handling and loading states

## 🎯 Usage

1. **Start the backend server** (port 8000)
2. **Start the React development server** (port 3000)
3. **Open the application** in your browser
4. **Follow the workflow**:
   - Start camera
   - Collect video context (optional)
   - Capture or upload final image
   - Generate Instagram caption

## 🚀 Deployment

The built application can be deployed to any static hosting service:

- Vercel
- Netlify
- AWS S3 + CloudFront
- GitHub Pages

## 🔄 Backend Integration

This frontend integrates with the existing FastAPI backend:
- Video context analysis endpoint
- Caption generation endpoint
- Health check endpoint

No changes to the backend are required - this is a complete frontend replacement.