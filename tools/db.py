"""数据库操作"""
import numpy as np
import pymysql
from tqdm import tqdm


class MyDB:
    def __init__(self, host='localhost', port=3306, user='root', database='face', password='123456'):
        try:
            self.db = pymysql.connect(
                host=host,
                port=port,
                user=user,
                database=database,
                password=password
            )
        except Exception as e:
            print(e)
            return
        self.cursor = self.db.cursor()
        print('数据连接成功')

    def insert_feature(self, name, feature):
        """
        输入feature为compute_face_descriptor()输出
        存入np.array().tobytes()类型
        """
        if name is None:
            print('姓名不能为空')
            return
        if feature is None:
            print('人脸特征不能为空')
        feature = np.array(feature).tobytes()
        sql = "insert into feature(name, feature) values(%s, %s)"
        self.cursor.execute(sql, (name, feature))
        self.db.commit()

    def get_feature(self, name):
        """
        获取个人的人脸特征
        :param name: 姓名
        :return: 返回np.array类型的feature
        """
        sql = "select feature from feature where name=%s"
        self.cursor.execute(sql, name)
        data_fetch = self.cursor.fetchall()
        return np.frombuffer(data_fetch[0][0])

    def load_all_feature(self):
        """
        加载数据库里所有人及其人脸特征
        :return: dict{name:feature}
        """
        sql = "select name, feature from feature"
        self.cursor.execute(sql)
        data_fetch = self.cursor.fetchall()
        ret = {}
        for d in tqdm(data_fetch):
            ret[d[0]] = np.frombuffer(d[1])
        return ret

    def __del__(self):
        self.db.close()
        print('数据库关闭')
