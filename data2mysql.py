import pymysql

class DataMemorizer(object):
    def __init__(self):
        self.database = pymysql.connect("localhost", "root", "root", "cqssc")
        self.cur = self.database.cursor()
 
    def createTable(self, year):
        '''
        在数据库中创建新表
        :param year: 年份，表的命名格式为“year+年份”，如year2018
        :return: 
        '''
        table_name = "year%s" % year
        table = self._toChar(table_name)
        if self._hasThisTable(table_name):
            # 如果这个表已存在，直接返回
            return table_name
        # 主键字段dates，为日期，存放格式为XXXX的月日4位字符串
        # reward存放当天一整天的开奖数据，是一个120*6的数组，数据类型设置为longblob，其实我觉得blob也可以
        sql = "create table " + table + "(dates char(4) not null primary key, reward longblob not null) ENGINE=myisam DEFAULT CHARSET=utf8;"
        self.cur.execute(sql)
        self.database.commit()
        print("创建新表：%s..." % table_name)
        return table_name
 
    def insertData(self, table_name, dateID, data):
        '''
        向数据库插入数据
        :param table_name:表名 
        :param dateID: 主键值
        :param data: 数据
        :return: 
        '''
        table = self._toChar(table_name)
        date = self._toStr(dateID)
        if self._hasThisId(table_name, dateID):
            # 如果这个ID已经存在，跳转到修改数据方法
            self.updateData(table_name, dateID, data)
        else:
            # 先要将numpy数组转换成二进制流，才能存到数据库中
            b_data = data.tostring()
            sql = "insert into " + table + " values(" + date + ", %s);"
            self.cur.execute(sql, (b_data,))
            self.database.commit()
            print("已插入数据：%s." % dateID)
 
    def updateData(self, table_name, dateID, data):
        '''
        更新数据库数据
        :param table_name:表名 
        :param dateID: 主键值
        :param data: 需要修改成的数据
        :return: 
        '''
        table = self._toChar(table_name)
        date = self._toStr(dateID)
        if not self._hasThisId(table_name, dateID):
            # 如果没有这个主键值，那还改个屁，直接返回
            return
        # 同样也是将data这个numpy数组转换一下成二进制流数据
        b_data = data.tostring()
        sql = "update " + table + " set reward = %s where dates = %s;"
        self.cur.execute(sql, (b_data, date))
        self.database.commit()
        print("已更新数据：%s..." % dateID)
 
    def _hasThisTable(self, table_name):
        '''
        判断是否存在此表
        :param table_name:表名 
        :return: True  or  False
        '''
        sql = "show tables;"
        self.cur.execute(sql)
        results = self.cur.fetchall()
        for r in results:
            if r[0] == table_name:
                return True
        else:
            return False
 
    def _hasThisId(self, table_name, dateID):
        '''
        判断在此表中是否已经有此主键
        :param table_name: 表名
        :param dateID: 主键值
        :return: True  or  False
        '''
        sql = "select dates from " + table_name + ";"
        self.cur.execute(sql)
        ids = self.cur.fetchall()
        for i in ids:
            if i[0] == dateID:
                return True
        else:
            return False
 
    def _toChar(self, string):
        '''
        为输入的字符串添加一对反引号，用于表名、字段名等对关键字的规避
        :param string: 
        :return: 
        '''
        return "`%s`" % string
 
    def _toStr(self, string):
        '''
        为输入的字符串添加一对单引号，用于数值处理，规避字符串拼接后原字符串暴露问题
        :param string: 
        :return: 
        '''
        return "'%s'" % string
 
    def __del__(self):
        '''
        临走之前记得关灯关电关空调，还有关闭数据库资源
        :return: 
        '''
        self.database.close()