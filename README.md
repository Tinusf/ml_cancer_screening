# ml_cancer_screening

# Virtualenv
In order to run this program, we would recommend to use a `virtualenv`. 

In order to create a virtualenv you type in this command: `virtualenv venv`.

Activate the virtualenv in linux by using the command `source venv/bin/activate`.

Activate the virtualenv in Windows using the command `venv/Scripts/activate`

If you are running Windows 

# Pip packages
If you want to install all the necessary packages run this command:

`pip install -r requirements.txt`


Data downloaded from https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000


# Website setup
A simple way to host the website using a Debian and Apache2.

* Install Apache2
* Install the required files using `yarn install` and then build using `yarn build`
* Copy the `build/` folder to `/var/www/html/` Using `cp . /var/www/html/ -r`.
* copy the files from `apache2_configs` to `/etc/apache2/sites-available`.
* Enable some modules and the config files using `sudo a2enmod proxy`, `sudo a2enmod proxy_http`, 
`sudo a2ensite tinusf.com.conf`.
* Disable the default configs: `sudo a2dissite 000-default-le-ssl.conf` and `sudo a2dissite 000-default.conf`. 
* Restart the apache2 service using `sudo systemctl restart apache2`.
* Create new A records on your DNS. I created `skincancer.tinusf.com` for the website and 
`skinflask.tinusf.com` for the flask backend. And redirected it to my server.
* Enable https on both the flask backend and frontend. Using 
`sudo certbot --apache -d skinflask.tinusf.com -d skincancer.tinusf.com`.


For the backend:
* Simply run the `./start_flask_api.sh` in a `screen`.

# TensorBoard
In order to visualize your training, you can have a look at the tensorboard by using the command  
`tensorboard --logdir logs/fit`.
