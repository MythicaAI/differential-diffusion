In order to setup the webservice, the following must be done in a local bash shell:

Open or create `/etc/systemd/system/inpaint.service` in a text editor and enter the following, edit accordingly and then save the file:

```
[Unit]
Description=Inpainting Service API
After=network.target

[Service]
User=your_user
Group=your_group
WorkingDirectory=/path/to/differential-diffusion/SD2
Environment="PATH=/path/to/differential-diffusion/venv/bin"
ExecStart=/path/to/differential-diffusion/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 run:app

[Install]
WantedBy=multi-user.target
```

the start the service and enable it to autostart:

```
sudo systemctl daemon-reload
sudo systemctl start inpaint
sudo systemctl enable inpaint
```

Install nginx:

`sudo apt install nginx`

Configure nginx:

```
sudo ln -s /path/to/differential-diffusion/SD2/nginx /etc/nginx/sites-available/inpaint
sudo ln -s /etc/nginx/sites-available/inpaint /etc/nginx/sites-enabled/inpaint
```

Run nginx:

```
sudo nginx -t
sudo systemctl restart nginx
```