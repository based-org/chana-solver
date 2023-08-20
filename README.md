# Chana's solver

It uses [Yukariin's](https://github.com/drunohazarb/4chan-captcha-solver/issues/1) (based on AUTOMATIC1111's code) model and implementation for solving 4chan's captchas.

You can easily host it now with the Dockerfile or just use systemd file after installing the dependencies from `requirements.txt` in a `venv`. If you use chana, just add your server ip to settings instead of the cloud one.



### Example usage
```
curl -X POST -F file="sneed.png" ${SERVER_IP}
```
