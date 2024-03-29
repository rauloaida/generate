const express = require('express');
const fs = require('fs');
const app = express();
const port = 3000;

app.use(express.raw({ type: 'image/png', limit: '10mb' }));

app.post('/save-image', (req, res) => {
  const imageBuffer = req.body;
  const filename = `image-${Date.now()}.png`;
  fs.writeFile(`./images/${filename}`, imageBuffer, 'binary', (err) => {
    if (err) {
      console.log(err);
      res.status(500).send('Error saving the image');
    } else {
      console.log(`Saved image as ${filename}`);
      res.status(200).send(`Saved image as ${filename}`);
    }
  });
});

app.listen(port, () => {
  console.log(`Image save server listening at http://localhost:${port}`);
});
