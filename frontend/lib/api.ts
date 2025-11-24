import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
});

export const uploadFile = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
};

export async function uploadSession(
    files: {
        lap_start: File;
        lap_end: File;
        lap_time: File;
        telemetry: File;
    },
    onProgress?: (progress: number) => void
) {
    const formData = new FormData();
    formData.append('lap_start', files.lap_start);
    formData.append('lap_end', files.lap_end);
    formData.append('lap_time', files.lap_time);
    formData.append('telemetry', files.telemetry);

    const response = await axios.post(`${API_BASE_URL}/upload-session`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
            if (onProgress && progressEvent.total) {
                const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                onProgress(percentCompleted);
            }
        },
    });

    return response;
}

export async function getSessionLaps(sessionId: string) {
    const response = await axios.get(`${API_BASE_URL}/session/${sessionId}/laps`);
    return response;
}

export async function getLapData(sessionId: string, lapNumber: number) {
    const response = await axios.get(`${API_BASE_URL}/session/${sessionId}/lap/${lapNumber}`);
    return response;
}

export async function getSessionMetadata(sessionId: string) {
    const response = await axios.get(`${API_BASE_URL}/session/${sessionId}/laps`);
    return response;
}

export async function getMistakeAnalysis(sessionId: string, lapNumber: number) {
    const response = await axios.get(`${API_BASE_URL}/session/${sessionId}/lap/${lapNumber}/mistake-analysis`);
    return response;
}

export default api;
