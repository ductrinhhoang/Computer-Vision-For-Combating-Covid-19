from datetime import datetime
import requests

def get_date():
    '''
        Get current date time and parse to dictionary
    '''
    date     = datetime.now()
    date_str = str(date.strftime('%H-%d-%m-%Y'))
    hour, day, month, year = date_str.split('-')
    return {'hour': hour, 'day': day, 'month': month, 'year': year}

def convert_url(list_urls):
    '''
        Get high-speed (local) url
    '''
    list_new_urls = []
    for path_img in list_urls:
        if "http" in path_img:
            list_new_urls.append(path_img)
        else:
            if path_img.startswith("/media"):
                image_name = path_img[path_img.rfind("/") + 1:-4] + "_index" + path_img[-4:]
                folder_im = path_img[:path_img.rfind("/")]
                fbid_img = folder_im[folder_im.rfind("/") + 1:]
                path_seaweed = "http://10.9.3.50:8888/autotag/facebook/{0}/image/{1}".format(fbid_img, image_name)
                list_new_urls.append(path_seaweed)
            elif path_img.startswith("/mnt"):
                image_name = path_img[path_img.rfind("/") + 1:-4] + "_index2" + path_img[-4:]
                folder_im = path_img[:path_img.rfind("/")]
                fbid_img = folder_im[folder_im.rfind("/") + 1:]
                path_seaweed = "http://10.9.3.50:8888/autotag/facebook/{0}/image/{1}".format(fbid_img, image_name)
                r = requests.get(path_seaweed)
                if not str(r.status_code).startswith("2"):
                    path_seaweed = "http://27.72.100.153:9573" + path_img
                list_new_urls.append(path_seaweed)
            else:
                list_new_urls.append(path_img)
    return list_new_urls
    
if __name__ == '__main__':
    print(get_date())