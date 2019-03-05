import json
import os

class store_id:
    def __init__(self,label,id):

        self.file_name=os.getcwd()+'/data.txt'
        self.label=label
        self.id=id
        self.content={}
        self.make_dir=False

    def check_presenc(self):
        with open (self.file_name,'r') as file:
            try:
                self.content=json.load(file)
                print("iam the content",self.content)
            except:
                with open(self.file_name,'w')as file:
                    json.dump({},file)

        if self.id not in self.content:
            return False
        else:
            return True
    def write_json(self):
        presence=self.check_presenc()
        print ("gg:",self.id)
        if presence==False:
            print("iam here")
            self.content.update({self.id:self.label})
            try:
                with open(self.file_name,'w')as file:
                    json.dump(self.content,file,indent=4,sort_keys=True)
            except:
                with open(self.file_name)as file:
                    json.dump({self.id:self.label},file,indent=4,sort_keys=True)
            self.make_dir=True
        else:
            print("not making directory")
            self.make_dir=False
        return 0
