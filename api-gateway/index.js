const express = require('express');
const axios = require('axios');
const cors = require('cors');
const app = express();

app.use(cors());
app.use(express.json());

app.post('/predict', async (req, res) => {
  try {
    const response = await axios.post('http://localhost:5001/predict', req.body);
    res.json(response.data);
  } catch (error) {
    res.status(500).send("Error in prediction");
  }
});

app.listen(3001, () => {
  console.log('Node.js server running on port 3001');
});
