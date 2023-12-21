# Chana's solver

It uses [Yukariin's](https://github.com/drunohazarb/4chan-captcha-solver/issues/1) (based on AUTOMATIC1111's code) and [Moffatman's](https://github.com/moffatman) "black spot" model update and implementation for solving 4chan's captchas.

You can easily host it now with the Dockerfile or just use systemd file after installing the dependencies from `requirements.txt` in a `venv`. If you use Chana, you can use your own server for captcha solving instead of default one (see in settings tab).



### Example usage
```
curl -X POST -F file="@sneed.png" ${SERVER_IP}/solve
```
