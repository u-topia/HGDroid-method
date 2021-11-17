- get_features_from_apk.py

  正则表达式说明：

  '\s'表示任意空字符串

  '+'表示前面的字符必须至少出现一次

  '*'表示之前字符出现0次或多次

  '?'表示之前的字符出现0次或1次

  ```python
  class SamliParser(ModuleBase){
  # 构造函数，将输入内容传入到类内变量中，包括APK路径，文件名后缀
  def __init__(self);
  
  # 判断当前是否找到所有后缀为suffer的文件
  def parse_location(self);
  
  '''
  输入要解析的文件名，解析指定的文件
  输入为文件名，并在过程中将结果保存到类内变量中
  逐行解析文件，首先提取文件所属class信息，保存到current_class中，并在classes列表中加入当前class信息；
  接着相同的方式解析'.super'（父类继承信息）,'.field'(提取静态字段与实例字段)，'const-string'(常量字符串)，'.method'（定义的方法），'invoke'(调用的方法)等信息，并保存到文件中
  '''
  def parse_file(self, filename);
  
  # 获取所有.smali文件的位置并进行遍历，获得结果
  def parse_location(self);
  
  # 正则表达式匹配"\.class\s+(?P<class>.*);",匹配到则返回获取的class名，否则返回空
  def is_class(self, line);
      
  # 正则表达式匹配"\.super\s+(?P<parent>.*);",匹配到则返回父类名称，否则返回为空
  def is_class_parent(self, line);
      
  # 正则表达式匹配"\.field\s+(?P<property>.*);",匹配到则返回字段定义信息，否则返回为空
  def is_class_property(self, line);
      
  # 正则表达式匹配"const-string\s+(?P<const>.*)",匹配到则返回字符串信息，否则返回为空
  def is_const_string(self, line);
  
  # 正则表达式匹配"\.method\s+(?P<method>.*)$",匹配到返回定义的函数名，否则返回为空
  def is_class_method(self, line);
      
  # 正则表达式匹配"invoke-\w+(?P<invoke>.*)",匹配到返回调用的信息，否则返回为空
  def is_method_call(self, line);
      
  '''
  输入的字符串为match.group('class')
  例：.class public final La;
  通过该函数将正则匹配到的字符串，拆分其中的内容，包括类名，包名，深度，类型（public static等），路径（self.current-path)，properties，const-string,methods，在接下来的函数中进行提取
  '''
  def extract_class(self, data);
      
  '''
  提取类属性的信息
  例：.field private static final a:Ljava/util/Map;
  首先通过“ ”分开字符串，接着获取名字和类型，通过：分割
  最后将前面的内容用空格组合得到字符串传入info
  '''
  def extract_class_property(self, data);
      
  '''
  提取静态字符串信息
  例：const-string v1, "/invalidRequest"
  再次对内容进行匹配：'(?P<var>.*),\s+"(?P<value>.*)"'
  将data分隔开为名称和值
  '''
  def extract_const_string(self, data);
  
  '''
  提取定义的方法的信息
  例：.method static constructor <clinit>()V
  对后面的内容进行匹配："(?P<name>.*)\((?<P<args>.*)\)(?P<return>.*)"
  分别对应名字，参数和返回值
  类型为前部分内容，并且定义了调用信息，保存之后在方法中调用的信息
  '''
  def extract_class_method(self, data);
      
  '''
  提取定义的方法中调用
  例：invoke-direct {v0}, Ljava/util/HashMap;-><init>()V
  '(?P<local_args>\{.*\}),\s+(?P<dst_class>.*);->' +
  '(?P<dst_method>.*)\((?P<dst_args>.*)\)(?P<return>.*)'
  提取内容包括本地参数，目的class，目的方法，调用时的参数，返回值
  '''
  def extract_method_call(self, data);    
  }
  ```

  

  最终信息以数组字典形式存储，classes保存了在该文件下所有遍历到的所有current_classes信息，结果见下图

  ![image-20210902172243278](/Users/liumengyao/Library/Application Support/typora-user-images/image-20210902172243278.png)

  

- get_HIN_graph.py

  用于将获得的结构构建异构图，进行下一步的学习

- get_realtion_from_API.py

  用于获取API文档中的API关联信息等内容，构建HIN中的部分信息

- logs.py

  日志文件，可以将运行结果输入到文件中进行保存与之后的测试、查找

- parsers.py

  清单文件，包含了一些普遍的函数定义

- jdsal

- 

