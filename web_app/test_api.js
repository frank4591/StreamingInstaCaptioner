// Test script to verify the backend API is working
const fetch = require('node-fetch');

async function testAPI() {
    try {
        console.log('Testing health endpoint...');
        const healthResponse = await fetch('http://localhost:8000/health');
        const healthData = await healthResponse.json();
        console.log('Health check:', healthData);

        console.log('Testing analyze_frames endpoint...');
        // Create a simple test image
        const canvas = require('canvas');
        const testCanvas = canvas.createCanvas(100, 100);
        const ctx = testCanvas.getContext('2d');
        ctx.fillStyle = 'red';
        ctx.fillRect(0, 0, 100, 100);
        ctx.fillStyle = 'white';
        ctx.font = '20px Arial';
        ctx.fillText('TEST', 20, 50);

        const imageData = testCanvas.toDataURL('image/jpeg');

        const analyzeResponse = await fetch('http://localhost:8000/analyze_frames', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                frames: [imageData]
            })
        });

        if (analyzeResponse.ok) {
            const analyzeData = await analyzeResponse.json();
            console.log('Analyze frames response:', analyzeData);
        } else {
            console.error('Analyze frames failed:', analyzeResponse.status, await analyzeResponse.text());
        }

    } catch (error) {
        console.error('Test failed:', error);
    }
}

testAPI();
