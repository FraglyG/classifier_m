Public facing API for some zero shot classification model. Originally used for [ButlerBot](https://butlerbot.net) but since sunset. I'll update this readme if I ever recall which zero-shot model this was built for, though it likely will work for other huggingface zero shot classification models due to a unified API.

This was built to be used within a Docker container with a volume attached on `/app/data` (assuming this app is mounted to `/app`)

Note, this isn't a production build since it was only used for testing purposes. I used it under a private network and only made it public facing through another app as a proxy. Authentication will likely be required for a production build unless a similar approach is used.
