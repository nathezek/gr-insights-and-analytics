
# Gazoo Insights & Analytics

Advanced telemetry analysis dashboard for race engineers. Identify mistakes, optimize lap times, and visualize performance with AI-powered insights.

✨ Technologies
- `React`
- `TypeScript`
- `Next.js`
- `Tailwind CSS`
- `Python`
- `FastAPI`
- `Pandas`
- `Scikit-learn`
- `Plotly`

🚀 Features
- **Telemetry Visualization**: Interactive charts for speed, throttle, brake, and steering data.
- **AI Mistake Detection**: Automatically identifies cornering errors and time loss.
- **Track Heatmaps**: Visualizes speed and mistakes directly on the track map.
- **Automated Coaching**: Provides specific advice for improving lap times based on data.

🎯 Who it is for
- **Race Engineers**: To quickly analyze driver performance and vehicle dynamics.
- **Drivers**: To understand where time is being lost and how to improve.
- **Data Analysts**: To process and visualize large datasets of telemetry.

📍 What I learned
- Processing high-frequency telemetry data for real-time web visualization.
- Integrating a Python data science backend with a modern Next.js frontend.
- Building interactive and performant charts using Plotly and React.
- Designing a dark-mode, data-heavy UI that remains readable and user-friendly.

🚦 Running the Project

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd gr-insights-and-analytics
   ```

2. **Backend Setup**
   ```bash
   # Install dependencies (using uv or pip)
   uv sync  # or pip install -r requirements.txt
   
   # Run the backend server
   uvicorn main:app --reload
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   
   # Install dependencies
   bun install
   
   # Run the development server
   bun dev
   ```

4. **Open the App**
   Open [http://localhost:3000](http://localhost:3000) in your browser.

🎞️ Preview

[https://github.com/user-



attachments/assets/your-video-id-here](https://github.com/user-attachments/assets/7c141523-6789-4e4b-a21c-c77bea1fb401)

<!-- For local viewing: -->
<video width="100%" controls>
  <source src="./frontend/public/video.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


