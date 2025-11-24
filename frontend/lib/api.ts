import axios from 'axios';

const api = axios.create({
    baseURL: 'http://127.0.0.1:8000',
});

export const uploadFile = (file: File, onProgress?: (progress: number) => void) => {
    const formData = new FormData();
    formData.append('file', file);

    return api.post('/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
            if (progressEvent.total && onProgress) {
                const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                onProgress(percentCompleted);
            }
        },
    });
};

export const uploadSession = (
    files: {
        lap_start: File;
        lap_end: File;
        lap_time: File;
        telemetry: File;
    },
    onProgress?: (progress: number) => void
) => {
    const formData = new FormData();
    formData.append('lap_start', files.lap_start);
    formData.append('lap_end', files.lap_end);
    formData.append('lap_time', files.lap_time);
    formData.append('telemetry', files.telemetry);

    return api.post('/upload-session', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
            if (progressEvent.total && onProgress) {
                const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                onProgress(percentCompleted);
            }
        },
    });
};

export const getSessionLaps = (sessionId: string) => {
    return api.get(`/session/${sessionId}/laps`);
};

export const getLapData = (sessionId: string, lapNumber: number) => {
    return api.get(`/session/${sessionId}/lap/${lapNumber}`);
};

export const getSessionMetadata = (sessionId: string) => {
    return api.get(`/session/${sessionId}/laps`);
};

export default api;
