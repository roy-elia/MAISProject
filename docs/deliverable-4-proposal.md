# Deliverable 4: Final Demonstration Proposal

## Final Product Description

We have built a **full-stack web application** that demonstrates our Reddit year prediction model. The application consists of:

1. **Interactive Demo Section**: Users can input Reddit-style comments and receive real-time predictions showing whether the text is from 2008-2010 or 2020-2022, along with confidence scores and class probabilities.

2. **Comprehensive Project Showcase**: A polished landing page featuring:
   - Hero section with project overview and key metrics (85% accuracy)
   - Dataset description and statistics
   - Methodology explanation (RoBERTa-base fine-tuning, 2-bin classification)
   - Results visualization (confusion matrix, prediction plots, error distributions)
   - Team members section
   - Conclusion and next steps

3. **Model Integration**: The trained RoBERTa-base 2-bin classifier (2008-2010 vs 2020-2022) is fully integrated and accessible through a REST API, with a regression fallback model for robustness.

---

## Technology Stack & Justification

### Frontend: Next.js 14 + TypeScript + Tailwind CSS

**Technologies:**
- **Next.js 14** (React framework)
- **TypeScript** (type safety)
- **Tailwind CSS** (utility-first styling)
- **Shadcn UI** (component library)
- **Lucide React** (icons)

**Justification:**
- **Next.js** was chosen for its excellent developer experience, built-in routing, API routes capability, and production-ready optimizations (SSR, static generation, image optimization)
- **TypeScript** ensures type safety and reduces runtime errors, especially important when integrating with the backend API
- **Tailwind CSS** enables rapid UI development with consistent design tokens and responsive utilities
- **Shadcn UI** provides accessible, customizable components that match modern design standards
- The combination allows for a professional, responsive website that works seamlessly across devices

### Backend: FastAPI + Python

**Technologies:**
- **FastAPI** (Python web framework)
- **Uvicorn** (ASGI server)
- **PyTorch** (deep learning framework)
- **Hugging Face Transformers** (pre-trained models)

**Justification:**
- **FastAPI** provides automatic API documentation (Swagger/OpenAPI), excellent performance, and native async support
- **Python** is the standard for ML/AI development, making it easy to integrate our PyTorch models
- **Hugging Face Transformers** provides pre-trained RoBERTa models and easy fine-tuning capabilities
- FastAPI's automatic request/response validation with Pydantic ensures type safety and clear error messages
- The async nature allows the API to handle multiple requests efficiently

### Model Serving Architecture

**Technologies:**
- **YearPredictor** class (custom inference wrapper)
- **Model loading with fallback** (classifier → regression)
- **REST API endpoint** (`/predict`)

**Justification:**
- Separating model inference into a `YearPredictor` class allows for clean abstraction and easy testing
- Implementing a fallback mechanism (classifier → regression) ensures the API remains functional even if classifier weights are missing
- REST API design is standard, easy to integrate with any frontend, and allows for future expansion (batch predictions, different model versions)

### Deployment & Sharing

**Technologies:**
- **ngrok** (tunneling service for local development)
- **Next.js API Routes** (proxy for backend)

**Justification:**
- **ngrok** allows us to share the local development environment without deploying to production servers
- **Next.js API Routes** act as a proxy, allowing the frontend to make relative API calls that work both locally and through ngrok
- This setup enables easy demos and presentations without the complexity of cloud deployment

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    User's Browser                            │
│  (Accesses via ngrok URL or localhost:3000)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ HTTP Requests
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Next.js Frontend (Port 3000)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  React Components:                                    │   │
│  │  - Hero, Demo, Visualizations, Methodology, etc.      │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  API Route: /api/predict (Next.js API Route)         │   │
│  │  - Proxies requests to FastAPI backend                │   │
│  └──────────────────────┬───────────────────────────────┘   │
└──────────────────────────┼───────────────────────────────────┘
                           │
                           │ HTTP POST /predict
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            FastAPI Backend (Port 8000)                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Endpoint: POST /predict                               │   │
│  │  - Receives: { text: string, task_type: string }      │   │
│  │  - Returns: { predicted_year_range, confidence, ... } │   │
│  └──────────────────────┬───────────────────────────────┘   │
│                         │                                    │
│  ┌──────────────────────▼───────────────────────────────┐   │
│  │  YearPredictor Class                                 │   │
│  │  - Loads model weights from disk                     │   │
│  │  - Tokenizes input text                              │   │
│  │  - Runs inference on RoBERTa model                   │   │
│  │  - Returns predictions                               │   │
│  └──────────────────────┬───────────────────────────────┘   │
└──────────────────────────┼───────────────────────────────────┘
                           │
                           │ Model Loading & Inference
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Model Files (On Disk)                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  backend/models/reddit_year_model/                    │   │
│  │  - config.json (model configuration)                  │   │
│  │  - model.safetensors (trained weights)                │   │
│  │  - tokenizer.json, vocab.json, etc.                   │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  backend/models/reddit_year_model_regressor/ (fallback)│   │
│  │  - Same structure as above                             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Input**: User types a Reddit comment in the demo textarea
2. **Frontend Request**: React component calls `/api/predict` (Next.js API route)
3. **API Proxy**: Next.js API route forwards request to `http://localhost:8000/predict`
4. **Backend Processing**: FastAPI receives request, validates input with Pydantic
5. **Model Inference**: `YearPredictor` loads model (if not already loaded), tokenizes text, runs inference
6. **Response**: Backend returns JSON with `predicted_year_range`, `confidence`, `class_probabilities`
7. **UI Update**: Frontend displays results with visual feedback (year range, confidence bars, probabilities)

---

## Integration Approach

### Model Loading Strategy

We implemented a **lazy loading** approach with fallback:

1. **Primary Model**: On first request, the backend attempts to load the 2-bin classifier from `./backend/models/reddit_year_model/`
2. **Fallback Model**: If classifier weights are missing, it automatically falls back to the regression model in `./backend/models/reddit_year_model_regressor/`
3. **Caching**: Once loaded, the model stays in memory for subsequent requests (singleton pattern)

This ensures:
- The API is always functional (graceful degradation)
- Fast response times after initial load
- Easy model updates (just restart the server)

### API Design

**Endpoint**: `POST /predict`

**Request**:
```json
{
  "text": "no cap this meme is wild fr",
  "task_type": "classification"
}
```

**Response**:
```json
{
  "predicted_year": 2021,
  "predicted_year_range": "2020-2022",
  "confidence": 0.87,
  "class_probabilities": {
    "2008-2010": 0.13,
    "2020-2022": 0.87
  }
}
```

**Error Handling**: The frontend includes fallback UI that shows placeholder results if the API is unavailable, ensuring the demo always works.

### Frontend-Backend Communication

- **Development**: Frontend uses Next.js API routes (`/api/predict`) that proxy to `localhost:8000`
- **Production Ready**: The proxy can be easily configured to point to a deployed backend URL via environment variables
- **CORS**: FastAPI backend includes CORS middleware to allow cross-origin requests

---

## Experience with Technologies

### Prior Experience

**Team members had experience with:**
- **Python**: Extensive experience with Python for data science and ML
- **React**: Some team members had prior React experience
- **PyTorch**: Experience from model training phase
- **Hugging Face**: Familiarity from using pre-trained models

**New Technologies Learned:**
- **Next.js**: No prior experience - learned through official documentation and tutorials
- **FastAPI**: Limited prior experience - learned through FastAPI documentation
- **TypeScript**: Some team members were new to TypeScript - learned through practice and TypeScript handbook
- **Tailwind CSS**: New technology - learned through Tailwind documentation and examples
- **Shadcn UI**: New component library - learned through their documentation

### Learning Resources

1. **Next.js Documentation**: https://nextjs.org/docs
   - Used for understanding App Router, API routes, and deployment
   
2. **FastAPI Tutorial**: https://fastapi.tiangolo.com/tutorial/
   - Used for learning request/response models, CORS, and async endpoints
   
3. **Tailwind CSS Documentation**: https://tailwindcss.com/docs
   - Used for learning utility classes and responsive design
   
4. **Shadcn UI Components**: https://ui.shadcn.com/
   - Used for learning component patterns and customization
   
5. **MAIS Workshop Recording**: 
   - Referenced for general ML model integration patterns
   - YouTube: https://www.youtube.com/watch?v=aucqOA6kyiU

6. **Online Tutorials**:
   - React + TypeScript tutorials for type-safe component development
   - FastAPI + React integration guides
   - Next.js deployment guides

### Challenges & Solutions

**Challenge 1: Model Loading in Production-like Environment**
- **Problem**: Ensuring models load correctly and handle missing files gracefully
- **Solution**: Implemented fallback mechanism and proper error handling in `YearPredictor` class

**Challenge 2: Frontend-Backend Communication**
- **Problem**: CORS issues and API URL configuration
- **Solution**: Added CORS middleware in FastAPI and created Next.js API proxy route for seamless integration

**Challenge 3: Type Safety Across Stack**
- **Problem**: Ensuring TypeScript types match Python Pydantic models
- **Solution**: Defined clear API contracts and used TypeScript interfaces that mirror Pydantic models

**Challenge 4: Responsive Design**
- **Problem**: Making the website look good on all screen sizes
- **Solution**: Used Tailwind's responsive utilities and tested on multiple devices

---

## Key Features Implemented

1. ✅ **Interactive Demo**: Real-time predictions with visual feedback
2. ✅ **Model Integration**: Full integration of trained RoBERTa model
3. ✅ **Fallback Mechanism**: Regression model fallback for robustness
4. ✅ **Visualizations**: Confusion matrix, prediction plots, error distributions
5. ✅ **Responsive Design**: Works on desktop, tablet, and mobile
6. ✅ **Dark Mode Support**: Theme toggle for user preference
7. ✅ **API Documentation**: FastAPI auto-generates Swagger docs at `/docs`
8. ✅ **Error Handling**: Graceful degradation when API is unavailable
9. ✅ **Shareable Link**: ngrok integration for easy demo sharing

---

## Future Enhancements (If Time Permits)

- Deploy to production (Vercel for frontend, Railway/Render for backend)
- Add batch prediction endpoint
- Implement model versioning
- Add user authentication for saving predictions
- Add historical prediction tracking
- Implement A/B testing for different model versions

---

## Conclusion

We successfully integrated our trained Reddit year prediction model into a polished, full-stack web application. The technology stack was chosen for its modern best practices, developer experience, and production-readiness. Through learning Next.js, FastAPI, and modern frontend tooling, we created a professional demonstration platform that effectively showcases our model's capabilities.

The application is ready for demonstration at the project fair and can be easily shared via ngrok or deployed to production hosting services.

---

## Repository

**GitHub Repository**: https://github.com/MarcoLipari/MAIS202-project

