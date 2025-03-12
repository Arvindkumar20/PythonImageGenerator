const express = require("express");
const { exec } = require("child_process");
const path = require("path");

const app = express();
const PORT = 5000;
app.use(express.json());

// Serve images statically
app.use("/images", express.static(path.join(__dirname, "images")));

app.post("/generate", (req, res) => {
    const { prompt } = req.body;
    if (!prompt) {
        return res.status(400).json({ error: "Prompt is required" });
    }

    // Run Python script to generate the image
    exec(`python img.py "${prompt}"`, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error: ${stderr}`);
            return res.status(500).json({ error: "Failed to generate image" });
        }

        const imagePath = stdout.trim();
        const imageUrl = `${req.protocol}://${req.get("host")}/images/${path.basename(imagePath)}`;
        return res.json({ imageUrl });
    });
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
