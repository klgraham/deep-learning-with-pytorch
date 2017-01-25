## Usage

#### Training
`$ python train.py <path to data>`

#### Prediction
`$ python predict.py <text to classify>`

#### Use the server
`$ python server.py` 

Hit the server at `http://localhost:8080/<text to classify>`

If the text has spaces, you'll want to use `%20` in place of the spaces.

When querying on the command line, then the spaces are okay. Just place the text in quotes.