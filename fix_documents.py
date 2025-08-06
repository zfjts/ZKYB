import os
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from main import Base, DocumentDB

# 配置数据库
SQLALCHEMY_DATABASE_URL = "sqlite:///d:/STU_ZF/auto_control_platform/database_new.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

db = SessionLocal()

try:
    # 获取documents目录中的所有文件
    docs_dir = os.path.join(os.path.dirname(__file__), "documents")
    files = os.listdir(docs_dir)
    print(f"documents目录中的文件数量: {len(files)}")
    
    # 查询数据库中已有的文档
    existing_docs = db.query(DocumentDB).all()
    existing_filenames = [doc.file_name for doc in existing_docs]
    print(f"数据库中已有的文档数量: {len(existing_docs)}")
    
    # 将未记录的文件添加到数据库
    added_count = 0
    for file_name in files:
        if file_name not in existing_filenames:
            file_path = os.path.join(docs_dir, file_name)
            # 猜测文件类型
            if file_name.endswith('.pdf'):
                file_type = 'application/pdf'
            elif file_name.endswith('.doc') or file_name.endswith('.docx'):
                file_type = 'application/msword'
            elif file_name.endswith('.xls') or file_name.endswith('.xlsx'):
                file_type = 'application/vnd.ms-excel'
            elif file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                file_type = 'image/jpeg'
            elif file_name.endswith('.png'):
                file_type = 'image/png'
            else:
                file_type = 'application/octet-stream'
            
            # 创建文档记录
            title = os.path.splitext(file_name)[0]
            # 简单分类
            if '规范' in title or '标准' in title or '规程' in title:
                category = '技术规范'
            elif '手册' in title:
                category = '操作手册'
            else:
                category = '其他'
            
            db_document = DocumentDB(
                title=title,
                file_name=file_name,
                file_path=file_path,
                file_type=file_type,
                category=category,
                upload_date=datetime.now(),
                equipment_id=None,
                description=f"自动修复添加的文档: {title}"
            )
            db.add(db_document)
            added_count += 1
    
    # 提交更改
    db.commit()
    print(f"成功添加到数据库的文档数量: {added_count}")

except Exception as e:
    db.rollback()
    print(f"处理出错: {str(e)}")
finally:
    db.close()