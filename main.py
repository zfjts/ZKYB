from fastapi import FastAPI, HTTPException, Depends, Request, Form, UploadFile, File, Response
from fastapi.responses import HTMLResponse, RedirectResponse, Response, FileResponse, JSONResponse
import pandas as pd
import os
from io import BytesIO
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Depends, File, Form, UploadFile, HTTPException
import os
import csv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
import uvicorn
import os

# 初始化FastAPI应用
app = FastAPI(title="自控设备全生命周期管理平台")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# 添加月份过滤器
from datetime import datetime

def add_months(date_str, months):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    month = date.month + months - 1
    year = date.year + month // 12
    month = month % 12 + 1
    day = min(date.day, [31, 29 if (year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month-1])
    return datetime(year, month, day).strftime('%Y-%m-%d')

templates.env.filters['add_months'] = add_months

@app.get("/@vite/client")
async def vite_client():
    return Response(status_code=204)

# 配置数据库
SQLALCHEMY_DATABASE_URL = "sqlite:///d:/STU_ZF/auto_control_platform/database_new.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 依赖项：获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 设备模型
class EquipmentDB(Base):
    __tablename__ = "equipment"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    model = Column(String)
    production_number = Column(String, unique=True, nullable=True)
    manufacturer = Column(String, nullable=True)
    installation_date = Column(DateTime, nullable=True)
    location = Column(String)
    status = Column(String)  # 运行状态
    equipment_type = Column(String)  # 设备类型
    working_life = Column(Float)  # 工作寿命(月)
    last_maintenance = Column(DateTime)
    next_maintenance = Column(DateTime)
    decommission_date = Column(DateTime, nullable=True)  # 退役日期
    parameters = Column(String, nullable=True)  # 存储参数的JSON字符串
    attachments = Column(String, nullable=True)  # 存储附属信息的JSON字符串





# 测试路由：添加测试设备
@app.get("/test/add_equipment/")
async def add_test_equipment(db: Session = Depends(get_db)):
    # 检查是否已存在测试设备
    existing = db.query(EquipmentDB).filter(EquipmentDB.name == "测试设备").first()
    if existing:
        return {"message": "测试设备已存在", "id": existing.id}

    # 创建测试设备
    test_equipment = EquipmentDB(
        name="测试设备",
        model="Model X",
        installation_date=datetime(2022, 1, 1),
        location="车间A",
        status="运行中",
        working_life=60,  # 5年
        last_maintenance=datetime(2023, 12, 1),
        next_maintenance=datetime(2024, 3, 1)
    )

    db.add(test_equipment)
    db.commit()
    db.refresh(test_equipment)
    return {"message": "测试设备已添加", "id": test_equipment.id}

# 设备参数路由
@app.get("/equipment/{equipment_id}/parameters")
async def equipment_parameters(request: Request, equipment_id: int, db: Session = Depends(get_db)):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")
    
    # 解析参数（假设存储为JSON字符串）
    import json
    parameters = json.loads(equipment.parameters) if equipment.parameters else {}
    
    # 删除模拟参数记录数据
    parameter_records = []
    
    return templates.TemplateResponse("equipment_parameters.html", {
        "request": request,
        "equipment": equipment,
        "parameters": parameters,
        "parameter_records": parameter_records
    })

# 添加参数记录路由
@app.post("/equipment/{equipment_id}/parameters/add")
async def add_parameter_record(
    equipment_id: int,
    param_name: str = Form(...),
    param_value: str = Form(...),
    db: Session = Depends(get_db)
):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")
    
    # 解析现有参数记录
    import json
    parameters = json.loads(equipment.parameters) if equipment.parameters else {}
    parameter_records = parameters.get("records", [])
    
    # 添加新参数记录
    new_record = {
        "param_name": param_name,
        "param_value": param_value,
        "modified_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    parameter_records.append(new_record)
    
    # 更新参数数据
    parameters["records"] = parameter_records
    equipment.parameters = json.dumps(parameters)
    db.commit()
    
    return RedirectResponse(f"/equipment/{equipment_id}/parameters", status_code=303)

# 设备附属信息路由
@app.get("/equipment/{equipment_id}/attachments")
async def equipment_attachments(request: Request, equipment_id: int, db: Session = Depends(get_db)):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")
    
    # 解析附属信息（假设存储为JSON字符串）
    import json
    attachments_data = json.loads(equipment.attachments) if equipment.attachments else {}
    
    # 从解析的数据中获取附属设备和文档
    attached_equipment = attachments_data.get("equipment", [])
    attached_documents = attachments_data.get("documents", [])
    
    attachments = {
        "equipment": attached_equipment,
        "documents": attached_documents
    }
    
    return templates.TemplateResponse("equipment_attachments.html", {
        "request": request,
        "equipment": equipment,
        "attachments": attachments
    })

# 添加附属设备路由
@app.post("/equipment/{equipment_id}/attachments/add_equipment")
async def add_attached_equipment(
    equipment_id: int,
    name: str = Form(...),
    model: str = Form(None),
    manufacturer: str = Form(None),
    relation_type: str = Form(...),
    db: Session = Depends(get_db)
):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")
    
    # 解析现有附属信息
    import json
    attachments_data = json.loads(equipment.attachments) if equipment.attachments else {}
    attached_equipment = attachments_data.get("equipment", [])
    
    # 添加新附属设备
    new_equipment = {
        "name": name,
        "model": model,
        "manufacturer": manufacturer,
        "relation_type": relation_type
    }
    attached_equipment.append(new_equipment)
    
    # 更新数据库
    attachments_data["equipment"] = attached_equipment
    equipment.attachments = json.dumps(attachments_data)
    db.commit()
    
    return RedirectResponse(f"/equipment/{equipment_id}/attachments", status_code=303)

# 删除附属设备路由
@app.get("/equipment/{equipment_id}/attachments/delete_equipment/{index}")
async def delete_attached_equipment(equipment_id: int, index: int, db: Session = Depends(get_db)):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")
    
    # 解析现有附属信息
    import json
    attachments_data = json.loads(equipment.attachments) if equipment.attachments else {}
    attached_equipment = attachments_data.get("equipment", [])
    
    # 检查索引是否有效
    if 0 <= index < len(attached_equipment):
        attached_equipment.pop(index)
        attachments_data["equipment"] = attached_equipment
        equipment.attachments = json.dumps(attachments_data)
        db.commit()
    
    return RedirectResponse(f"/equipment/{equipment_id}/attachments", status_code=303)

# 添加附属文档路由
@app.post("/equipment/{equipment_id}/attachments/add_document")
async def add_attached_document(
    equipment_id: int,
    doc_name: str = Form(...),
    doc_type: str = Form(...),
    doc_description: str = Form(None),
    db: Session = Depends(get_db)
):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")

    # 解析现有的attachments数据
    attachments = {}
    if equipment.attachments:
        try:
            attachments = json.loads(equipment.attachments)
        except json.JSONDecodeError:
            pass

    # 确保documents数组存在
    if 'documents' not in attachments:
        attachments['documents'] = []

    # 添加新文档
    new_doc = {
        'id': len(attachments['documents']) + 1,
        'name': doc_name,
        'type': doc_type,
        'description': doc_description,
        'added_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    attachments['documents'].append(new_doc)

    # 更新设备的attachments字段
    equipment.attachments = json.dumps(attachments)
    db.commit()

    # 同时创建DocumentDB记录
    db_document = DocumentDB(
        title=doc_name,
        file_name=f"{doc_name}.{doc_type.lower()}",
        file_path="",  # 实际应用中应该上传文件并保存路径
        file_type=f"document/{doc_type.lower()}",
        category="设备附属文档",
        upload_date=datetime.now(),
        equipment_id=equipment_id,
        description=doc_description
    )
    db.add(db_document)
    db.commit()

    return RedirectResponse(url=f"/equipment/{equipment_id}/attachments", status_code=303)

@app.get("/equipment/{equipment_id}/attachments/delete_document/{doc_id}")
async def delete_attached_document(equipment_id: int, index: int, db: Session = Depends(get_db)):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")
    
    # 解析现有附属信息
    import json
    attachments_data = json.loads(equipment.attachments) if equipment.attachments else {}
    attached_documents = attachments_data.get("documents", [])
    
    # 检查索引是否有效
    if 0 <= index < len(attached_documents):
        attached_documents.pop(index)
        attachments_data["documents"] = attached_documents
        equipment.attachments = json.dumps(attachments_data)
        db.commit()
    
    return RedirectResponse(f"/equipment/{equipment_id}/attachments", status_code=303)

# 批量上传相关路由
@app.get("/equipment/batch/upload/")
async def batch_upload_equipment_form(request: Request):
    return templates.TemplateResponse("equipment_batch_form.html", {"request": request})

@app.post("/equipment/batch/upload/")
async def batch_upload_equipment(
    request: Request,
    file: UploadFile = File(...),
    skip_errors: bool = Form(False),
    override_existing: bool = Form(False),
    db: Session = Depends(get_db)
):
    # 验证文件类型
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="请上传Excel文件 (.xlsx)")

    try:
        # 读取Excel文件
        df = pd.read_excel(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件读取失败: {str(e)}")

    # 验证必要列
    required_columns = ['name', 'model', 'location', 'status']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise HTTPException(status_code=400, detail=f"缺少必要列: {', '.join(missing_columns)}")

    # 初始化结果统计
    success_count = 0
    error_count = 0
    errors = []

    # 处理每一行数据
    for index, row in df.iterrows():
        row_num = index + 2  # 行号从2开始（第一行是表头）
        try:
            # 验证状态值
            if row['status'] not in ['未安装', '运行中']:
                raise ValueError("状态必须是'未安装'或'运行中'")

            # 转换日期
            installation_date = None
            if 'installation_date' in row and pd.notna(row['installation_date']) and str(row['installation_date']).strip():
                try:
                    installation_date = datetime.strptime(str(row['installation_date']).strip(), "%Y-%m-%d")
                except ValueError:
                    raise ValueError("安装日期格式不正确，请使用YYYY-MM-DD格式")

            # 检查生产编号是否已存在
            production_number = row.get('production_number')
            existing_equipment = None

            if production_number:
                existing_equipment = db.query(EquipmentDB).filter(EquipmentDB.production_number == production_number).first()

            # 如果存在且不覆盖，则跳过
            if existing_equipment and not override_existing:
                error_count += 1
                errors.append({
                    'row': row_num,
                    'data': {'name': row['name'], 'model': row['model']},
                    'error': f"生产编号'{production_number}'已存在，且未选择覆盖"
                })
                continue

            # 计算下次维护时间（假设安装后30天首次维护）
            next_maintenance = None
            if installation_date:
                next_maintenance = installation_date.replace(day=min(installation_date.day + 30, 28))

            if existing_equipment:
                # 更新现有设备
                existing_equipment.name = row['name']
                existing_equipment.model = row['model']
                existing_equipment.manufacturer = row.get('manufacturer')
                existing_equipment.installation_date = installation_date
                existing_equipment.location = row['location']
                existing_equipment.status = row['status']
                existing_equipment.equipment_type = row.get('equipment_type')
                existing_equipment.next_maintenance = next_maintenance
                db.commit()
            else:
                # 创建新设备
                db_equipment = EquipmentDB(
                    name=row['name'],
                    model=row['model'],
                    production_number=production_number if pd.notna(production_number) else None,
                    manufacturer=row['manufacturer'],
                    installation_date=installation_date,
                    location=row['location'],
                    status=row['status'],
                    equipment_type=row.get('equipment_type'),
                    next_maintenance=next_maintenance
                )
                db.add(db_equipment)
                db.commit()

            success_count += 1
        except Exception as e:
            error_count += 1
            errors.append({
                'row': row_num,
                'data': {'name': row.get('name', '未知'), 'model': row.get('model', '未知')},
                'error': str(e)
            })
            if not skip_errors:
                raise HTTPException(status_code=400, detail=f"第{row_num}行数据错误: {str(e)}")

    # 提交所有更改
    db.commit()

    # 返回结果页面
    return templates.TemplateResponse("equipment_batch_result.html", {
        "request": request,
        "success_count": success_count,
        "error_count": error_count,
        "errors": errors
    })

# 模板下载路由
@app.get("/equipment/template/download/")
async def download_equipment_template():
    # 创建模板数据
    template_data = [
        {'name': '设备名称', 'model': '型号', 'production_number': '生产编号(可选)', 'manufacturer': '制造商', 'installation_date': '安装日期(YYYY-MM-DD)', 'location': '位置', 'status': '状态(未安装/运行中)'},
        {'name': '示例设备1', 'model': 'Model-X', 'production_number': 'SN123456', 'manufacturer': '示例厂商', 'installation_date': '2023-01-15', 'location': '车间A', 'status': '运行中'},
        {'name': '示例设备2', 'model': 'Model-Y', 'production_number': '', 'manufacturer': '示例厂商', 'installation_date': '2023-02-20', 'location': '车间B', 'status': '未安装'}
    ]

    # 创建Excel文件
    df = pd.DataFrame(template_data)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='设备模板')

    output.seek(0)

    # 设置缓存控制头，防止浏览器缓存旧版本
    headers = {
        'Content-Disposition': 'attachment; filename="equipment_template.xlsx"',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    }

    return Response(content=output.getvalue(), headers=headers, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


    

# 数据导出路由
@app.get("/equipment/export/")
async def export_equipment_data(db: Session = Depends(get_db)):
    # 查询所有设备数据
    equipments = db.query(EquipmentDB).all()

    # 创建数据列表
    data = []
    for eq in equipments:
        data.append({
            'id': eq.id,
            'name': eq.name,
            'model': eq.model,
            'production_number': eq.production_number if eq.production_number else '',
            'manufacturer': eq.manufacturer,
            'installation_date': eq.installation_date.strftime('%Y-%m-%d'),
            'location': eq.location,
            'status': eq.status,
            'next_maintenance': eq.next_maintenance.strftime('%Y-%m-%d') if eq.next_maintenance else ''
        })

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 创建临时文件
    temp_dir = os.path.join(os.path.dirname(__file__), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    temp_file = os.path.join(temp_dir, f"equipment_data_{timestamp}.xlsx")

    # 写入Excel文件
    df.to_excel(temp_file, index=False)

    # 设置缓存控制头，防止浏览器缓存旧版本
    headers = {
        'Content-Disposition': f'attachment; filename="equipment_data_{datetime.now().strftime('%Y%m%d')}.xlsx"',
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
    }

    return FileResponse(temp_file, headers=headers, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# 添加自定义过滤器到Jinja2模板环境
def add_months_filter(date_str, months):
    # 将字符串日期转换为datetime对象
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    # 添加指定的月数
    new_date = date_obj + relativedelta(months=months)
    # 格式化为字符串返回
    return new_date.strftime("%Y-%m-%d")

# 注册过滤器
templates.env.filters['add_months'] = add_months_filter

# 维护记录模型
class MaintenanceDB(Base):
    __tablename__ = "maintenance"

    id = Column(Integer, primary_key=True, index=True)
    equipment_id = Column(Integer, ForeignKey("equipment.id"))
    maintenance_date = Column(DateTime)
    maintenance_type = Column(String)
    performed_by = Column(String)
    notes = Column(String)


# 文档模型
class DocumentDB(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    file_name = Column(String)
    file_path = Column(String)
    file_type = Column(String)
    category = Column(String)
    upload_date = Column(DateTime)
    equipment_id = Column(Integer, ForeignKey("equipment.id"), nullable=True)
    description = Column(String)


# 创建数据库表
Base.metadata.create_all(bind=engine)

# Pydantic模型用于请求和响应
class EquipmentBase(BaseModel):
    name: str
    model: str
    production_number: str | None = None
    manufacturer: str
    installation_date: datetime
    location: str
    status: str
    equipment_type: str

class EquipmentCreate(EquipmentBase):
    pass

class Equipment(EquipmentBase):
    id: int
    last_maintenance: datetime | None
    next_maintenance: datetime | None
    decommission_date: datetime | None
    current_status: str
    equipment_type: str

    class Config:
        from_attributes = True

class MaintenanceReminder(BaseModel):
    equipment_id: int
    equipment_name: str
    model: str
    next_maintenance: datetime
    remaining_days: int
    current_status: str

    class Config:
        from_attributes = True

class DecommissionResponse(BaseModel):
    equipment_id: int
    equipment_name: str
    decommission_date: datetime
    status: str

    class Config:
        from_attributes = True

class MaintenanceBase(BaseModel):
    maintenance_date: datetime
    maintenance_type: str
    performed_by: str
    notes: str


class MaintenanceCreate(MaintenanceBase):
    equipment_id: int

class Maintenance(MaintenanceBase):
    id: int
    equipment_id: int

    class Config:
        from_attributes = True

# 文档Pydantic模型
class DocumentBase(BaseModel):
    title: str
    file_name: str
    file_type: str
    category: str
    description: str | None = None
    equipment_id: int | None = None

class DocumentCreate(DocumentBase):
    pass

class Document(DocumentBase):
    id: int
    file_path: str
    upload_date: datetime

    class Config:
        from_attributes = True

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, db: Session = Depends(get_db)):
    # 获取设备总数
    total_equipment = db.query(EquipmentDB).count()
    
    # 获取本月维护记录
    today = date.today()
    start_of_month = date(today.year, today.month, 1)
    monthly_maintenance = db.query(MaintenanceDB).filter(
        MaintenanceDB.maintenance_date >= start_of_month
    ).count()
    
    # 获取待维护设备 (假设30天内需要维护的设备)
    future_maintenance_date = today + timedelta(days=30)
    pending_equipment_ids = db.query(EquipmentDB.id).filter(
        EquipmentDB.next_maintenance <= future_maintenance_date,
        EquipmentDB.decommission_date == None
    ).distinct()
    pending_maintenance = pending_equipment_ids.count()
    
    # 获取文档总数
    total_documents = db.query(DocumentDB).count()
    
    # 获取设备状态分布
    status_distribution = {
        "运行中": db.query(EquipmentDB).filter(EquipmentDB.status == "运行中").count(),
        "即将过期": db.query(EquipmentDB).filter(EquipmentDB.status == "即将过期").count(),
        "已过期": db.query(EquipmentDB).filter(EquipmentDB.status == "已过期").count(),
        "停用": db.query(EquipmentDB).filter(EquipmentDB.status == "停用").count(),
        "未安装": db.query(EquipmentDB).filter(EquipmentDB.status == "未安装").count(),
    }
    
    # 获取近6个月维护记录
    maintenance_history = []
    for i in range(6):
        month_date = date(today.year, today.month - i, 1) if today.month - i > 0 else date(today.year - 1, 12 + (today.month - i), 1)
        next_month = date(month_date.year, month_date.month + 1, 1) if month_date.month < 12 else date(month_date.year + 1, 1, 1)
        count = db.query(MaintenanceDB).filter(
            MaintenanceDB.maintenance_date >= month_date,
            MaintenanceDB.maintenance_date < next_month
        ).count()
        month_name = month_date.strftime("%Y-%m")
        maintenance_history.append({"month": month_name, "count": count})
    
    # 按照月份升序排列
    maintenance_history.sort(key=lambda x: x["month"])
    
    # 将数据转换为JSON字符串，确保前端可以正确解析
    import json
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "total_equipment": json.dumps(total_equipment),
            "monthly_maintenance": json.dumps(monthly_maintenance),
            "pending_maintenance": json.dumps(pending_maintenance),
            "total_documents": json.dumps(total_documents),
            "status_distribution": json.dumps(status_distribution),
            "maintenance_history": json.dumps(maintenance_history)
        }
    )

@app.get("/equipment/", response_class=HTMLResponse)
async def read_equipment(request: Request, db: Session = Depends(get_db)):
    equipment = db.query(EquipmentDB).all()
    return templates.TemplateResponse("equipment_list.html", {"request": request, "equipment": equipment})

@app.get("/equipment/{equipment_id}", response_class=HTMLResponse)
async def read_equipment_detail(request: Request, equipment_id: int, db: Session = Depends(get_db)):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if equipment is None:
        raise HTTPException(status_code=404, detail="设备未找到")
    maintenance = db.query(MaintenanceDB).filter(MaintenanceDB.equipment_id == equipment_id).all()
    return templates.TemplateResponse("equipment_detail.html", {"request": request, "equipment": equipment, "maintenance": maintenance})

@app.get("/equipment/create/", response_class=HTMLResponse)
async def create_equipment_form(request: Request):
    return templates.TemplateResponse("equipment_form.html", {"request": request})

@app.post("/equipment/create/")
async def create_equipment(
    name: str = Form(...),
    model: str = Form(...),
    production_number: str = Form(None),
    manufacturer: str = Form(...),
    installation_date: str = Form(...),
    location: str = Form(...),
    status: str = Form(...),
    equipment_type: str = Form(...),
    working_life: int = Form(None),
    db: Session = Depends(get_db)
):
    # 转换日期字符串为datetime对象
    try:
        installation_date = datetime.strptime(installation_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式不正确，请使用YYYY-MM-DD格式")

    # 根据设备类型自动设置工作寿命
    if working_life is None:
        working_life = WORKING_LIFE_MAP.get(equipment_type, 60)

    # 计算下次维护时间（假设安装后30天首次维护）
    next_maintenance = None
    if installation_date:
        next_maintenance = installation_date.replace(day=min(installation_date.day + 30, 28))

    db_equipment = EquipmentDB(
            name=name,
            model=model,
            production_number=production_number,
            manufacturer=manufacturer,
            installation_date=installation_date,
            location=location,
            status=status,
            equipment_type=equipment_type,
            working_life=working_life,
            next_maintenance=next_maintenance
        )
    db.add(db_equipment)
    db.commit()
    db.refresh(db_equipment)
    return RedirectResponse(url=f"/equipment/{db_equipment.id}", status_code=303)

# 设备类型与工作寿命映射表 (月)
WORKING_LIFE_MAP = {
    'DCS': 240,
    'PLC': 180,
    '变送器': 120,
    '热电阻': 120,
    '热电偶': 60,
    '调节阀': 120,
    '切断阀': 180,
    '物位计': 180,
    '压力表': 60,
    '双金属温度计': 120
}

# 设备管理相关路由
@app.get("/equipment/{equipment_id}/edit", response_class=HTMLResponse)
async def edit_equipment_form(request: Request, equipment_id: int, db: Session = Depends(get_db)):
    print(f"访问编辑表单路由: /equipment/{equipment_id}/edit")
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if equipment is None:
        raise HTTPException(status_code=404, detail="设备未找到")
    return templates.TemplateResponse("equipment_form.html", {"request": request, "equipment": equipment})

@app.post("/equipment/{equipment_id}/edit")
async def update_equipment_form(
    equipment_id: int,
    name: str = Form(...),
    model: str = Form(...),
    production_number: str = Form(None),
    manufacturer: str = Form(...),
    installation_date: str = Form(...),
    location: str = Form(...),
    status: str = Form(...),
    equipment_type: str = Form(...),
    working_life: int = Form(None),
    db: Session = Depends(get_db)
):
    print(f"处理更新设备请求: /equipment/{equipment_id}/edit")
    db_equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if db_equipment is None:
        raise HTTPException(status_code=404, detail="设备未找到")

    # 转换日期字符串为datetime对象
    try:
        installation_date = datetime.strptime(installation_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式不正确，请使用YYYY-MM-DD格式")

    # 更新设备信息
    db_equipment.name = name
    db_equipment.model = model
    # 检查生产编号唯一性
    # 处理空值或"None"字符串的情况
    print(f"接收到的生产编号: {production_number}, 类型: {type(production_number)}")
    if production_number in (None, '', 'None'):
        db_equipment.production_number = None
        print("生产编号为空，设置为None")
    else:
        existing_equipment = db.query(EquipmentDB).filter(
            EquipmentDB.production_number == production_number,
            EquipmentDB.id != equipment_id
        ).first()
        if existing_equipment:
            raise HTTPException(status_code=400, detail=f"生产编号'{production_number}'已存在，请使用其他编号")
        db_equipment.production_number = production_number
    db_equipment.manufacturer = manufacturer
    db_equipment.installation_date = installation_date
    db_equipment.location = location
    db_equipment.status = status
    db_equipment.equipment_type = equipment_type
    # 根据设备类型自动设置工作寿命
    if working_life is None:
        working_life = WORKING_LIFE_MAP.get(equipment_type, 60)
    db_equipment.working_life = working_life

    db.commit()
    db.refresh(db_equipment)
    return RedirectResponse(url=f"/equipment/{equipment_id}", status_code=303)

@app.post("/equipment/{equipment_id}/delete")
async def delete_equipment(equipment_id: int, db: Session = Depends(get_db)):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if not equipment:
        raise HTTPException(status_code=404, detail="设备未找到")
    
    db.delete(equipment)
    db.commit()
    return RedirectResponse("/equipment/", status_code=303)

@app.put("/equipment/{equipment_id}/")
async def update_equipment(equipment_id: int, equipment: EquipmentCreate, db: Session = Depends(get_db)):
    db_equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if db_equipment is None:
        raise HTTPException(status_code=404, detail="设备未找到")
    # 更新设备信息
    for key, value in equipment.dict(exclude_unset=True).items():
        setattr(db_equipment, key, value)
    
    # 根据设备类型自动设置工作寿命（如果未提供）
    if db_equipment.working_life is None:
        db_equipment.working_life = WORKING_LIFE_MAP.get(db_equipment.equipment_type, 60)
    db.commit()
    db.refresh(db_equipment)
    return db_equipment

@app.put("/equipment/{equipment_id}/decommission/")
async def decommission_equipment(equipment_id: int, decommission_date: datetime = None, db: Session = Depends(get_db)) -> DecommissionResponse:
    """将设备标记为退役"""
    db_equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if db_equipment is None:
        raise HTTPException(status_code=404, detail="设备未找到")

    if db_equipment.decommission_date:
        raise HTTPException(status_code=400, detail="设备已停用")

    # 如果未提供退役日期，使用当前日期
    if not decommission_date:
        decommission_date = datetime.now()

    db_equipment.decommission_date = decommission_date
    db_equipment.status = "停用"
    db.commit()
    db.refresh(db_equipment)

    return {
        "equipment_id": db_equipment.id,
        "equipment_name": db_equipment.name,
        "decommission_date": db_equipment.decommission_date,
        "status": db_equipment.status
    }

@app.get("/maintenance/create/{equipment_id}", response_class=HTMLResponse)
async def create_maintenance_form(request: Request, equipment_id: int, db: Session = Depends(get_db)):
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if equipment is None:
        raise HTTPException(status_code=404, detail="设备未找到")
    return templates.TemplateResponse("maintenance_form.html", {"request": request, "equipment": equipment})

@app.get("/maintenance/create/")
async def maintenance_create_redirect(request: Request):
    return RedirectResponse(url="/equipment/", status_code=303)

@app.post("/maintenance/create/")
async def create_maintenance(
    equipment_id: int = Form(...),
    maintenance_date: str = Form(...),
    maintenance_type: str = Form(...),
    performed_by: str = Form(...),
    notes: str = Form(...),

    db: Session = Depends(get_db)
):
    # 转换日期字符串为datetime对象
    try:
        maintenance_date = datetime.strptime(maintenance_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式不正确，请使用YYYY-MM-DD格式")

    # 更新设备的最后维护时间和下次维护时间
    equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    if equipment is None:
        raise HTTPException(status_code=404, detail="设备未找到")

    equipment.last_maintenance = maintenance_date
    # 假设每3个月维护一次
    if equipment.next_maintenance is None or maintenance_date > equipment.next_maintenance:
        if maintenance_date:
            equipment.next_maintenance = maintenance_date.replace(month=min(maintenance_date.month + 3, 12))
            if equipment.next_maintenance.month < maintenance_date.month:
                equipment.next_maintenance = equipment.next_maintenance.replace(year=equipment.next_maintenance.year + 1)

    db_maintenance = MaintenanceDB(
        equipment_id=equipment_id,
        maintenance_date=maintenance_date,
        maintenance_type=maintenance_type,
        performed_by=performed_by,
        notes=notes,

    )
    db.add(db_maintenance)
    db.commit()
    db.refresh(db_maintenance)
    return RedirectResponse(url=f"/equipment/{equipment_id}", status_code=303)


# 维护提醒相关路由
@app.get("/maintenance/reminders/")
async def get_maintenance_reminders(request: Request, days: int = 30, db: Session = Depends(get_db)):
    """获取未来指定天数内需要维护的设备并渲染到模板"""
    today = datetime.now().date()
    future_date = today + timedelta(days=days)

    # 查询未来指定天数内需要维护的设备
    query = db.query(EquipmentDB)
    query = query.filter(EquipmentDB.next_maintenance <= future_date)
    query = query.filter(EquipmentDB.decommission_date == None)
    equipments = query.all()

    reminders = []
    for equipment in equipments:
        # 计算剩余天数
        remaining_days = (equipment.next_maintenance.date() - today).days
        reminders.append({
            "equipment_id": equipment.id,
            "equipment_name": equipment.name,
            "model": equipment.model,
            "next_maintenance": equipment.next_maintenance,
            "remaining_days": remaining_days,
            "current_status": equipment.current_status
        })

    # 按剩余天数排序
    reminders.sort(key=lambda x: x["remaining_days"])

    # 计算统计数据
    total_reminders = len(reminders)
    urgent_reminders = len([r for r in reminders if r['remaining_days'] <= 7])
    percent_in_7_days = round((urgent_reminders / total_reminders) * 100) if total_reminders > 0 else 0

    # 查询已完成的维护记录数
    completed_maintenances = db.query(MaintenanceDB).count()
    # 计算完成率
    total_equipment = db.query(EquipmentDB).count()
    completion_rate = round((completed_maintenances / total_equipment) * 100) if total_equipment > 0 else 0

    # 渲染模板
    return templates.TemplateResponse(
        "maintenance_reminders.html",
        {
            "request": request,
            "reminders": reminders,
            "total_reminders": total_reminders,
            "urgent_reminders": urgent_reminders,
            "percent_in_7_days": percent_in_7_days,
            "completed_maintenances": completed_maintenances,
            "completion_rate": completion_rate
        }
    )

# 文档管理API路由
@app.get("/documents/", response_class=HTMLResponse)
async def read_documents(request: Request, category: str = None, equipment_id: int = None, db: Session = Depends(get_db)):
    query = db.query(DocumentDB)
    if category:
        query = query.filter(DocumentDB.category == category)
    if equipment_id:
        query = query.filter(DocumentDB.equipment_id == equipment_id)
    documents = query.all()
    return templates.TemplateResponse("document_list.html", {"request": request, "documents": documents, "category": category, "equipment_id": equipment_id})

@app.get("/documents/create/", response_class=HTMLResponse)
async def create_document_form(request: Request, equipment_id: int = None, db: Session = Depends(get_db)):
    equipment = None
    if equipment_id:
        equipment = db.query(EquipmentDB).filter(EquipmentDB.id == equipment_id).first()
    equipment_list = db.query(EquipmentDB).all()
    return templates.TemplateResponse("document_form.html", {"request": request, "equipment": equipment, "equipment_list": equipment_list})

@app.post("/documents/create/")
async def create_document(
    request: Request,
    title: str = Form(...),
    category: str = Form(...),
    description: str | None = Form(None),
    equipment_id: int | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        # 创建文档目录（如果不存在）
        documents_dir = os.path.join(os.path.dirname(__file__), "documents")
        os.makedirs(documents_dir, exist_ok=True)

        # 保存文件
        file_path = os.path.join(documents_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        print(f"文件已保存: {file_path}")
    except Exception as e:
        print(f"文件保存失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")

    # 创建文档记录
    db_document = DocumentDB(
        title=title,
        file_name=file.filename,
        file_path=file_path,
        file_type=file.content_type,
        category=category,
        upload_date=datetime.now(),
        equipment_id=equipment_id,
        description=description
    )
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return RedirectResponse(url="/documents/", status_code=303)

@app.get("/documents/{document_id}", response_class=HTMLResponse)
async def read_document_detail(request: Request, document_id: int, db: Session = Depends(get_db)):
    document = db.query(DocumentDB).filter(DocumentDB.id == document_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail="文档未找到")
    equipment = None
    if document.equipment_id:
        equipment = db.query(EquipmentDB).filter(EquipmentDB.id == document.equipment_id).first()
    return templates.TemplateResponse("document_detail.html", {"request": request, "document": document, "equipment": equipment})

@app.post("/documents/delete/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    document = db.query(DocumentDB).filter(DocumentDB.id == document_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail="文档未找到")
    # 删除文件
    if os.path.exists(document.file_path):
        os.remove(document.file_path)
    # 删除数据库记录
    db.delete(document)
    db.commit()
    return RedirectResponse(url="/documents/", status_code=303)

@app.get("/documents/{document_id}/edit", response_class=HTMLResponse)
async def edit_document_form(request: Request, document_id: int, db: Session = Depends(get_db)):
    document = db.query(DocumentDB).filter(DocumentDB.id == document_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail="文档未找到")
    equipment_list = db.query(EquipmentDB).all()
    return templates.TemplateResponse("document_form.html", {"request": request, "document": document, "equipment_list": equipment_list})

@app.post("/documents/{document_id}/edit")
async def update_document(
    request: Request,
    document_id: int,
    title: str = Form(...),
    category: str = Form(...),
    description: str | None = Form(None),
    equipment_id: str | None = Form(None),
    db: Session = Depends(get_db)
):
    document = db.query(DocumentDB).filter(DocumentDB.id == document_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail="文档未找到")
    
    # 处理equipment_id
    try:
        equip_id = int(equipment_id) if equipment_id else None
    except ValueError:
        equip_id = None
    
    # 更新文档信息
    document.title = title
    document.category = category
    document.description = description
    document.equipment_id = equip_id
    document.update_date = datetime.now()
    
    db.commit()
    db.refresh(document)
    return RedirectResponse(url=f"/documents/{document_id}", status_code=303)

# 测试文档数据路由
@app.get("/test_documents/")
async def get_documents_count(db: Session = Depends(get_db)):
    # 查询文档总数
    documents_count = db.query(DocumentDB).count()
    return JSONResponse(content={"documents_count": documents_count})

# 文档查看路由
@app.get("/documents/view/{document_id}")
async def view_document(document_id: int, db: Session = Depends(get_db)):
    document = db.query(DocumentDB).filter(DocumentDB.id == document_id).first()
    if document is None:
        raise HTTPException(status_code=404, detail="文档未找到")
    
    # 检查文件路径和存在性
    print(f"尝试查看文档: {document.title}, 文件路径: {document.file_path}")
    if not os.path.exists(document.file_path):
        print(f"文件不存在: {document.file_path}")
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # 确保文件可读
    try:
        with open(document.file_path, 'rb') as f:
            pass
        print(f"文件可读: {document.file_path}")
    except Exception as e:
        print(f"文件读取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"文件读取失败: {str(e)}")
    
    if document.file_type.startswith("application/pdf") or document.file_type.startswith("image/"):
        print(f"返回文件内容: {document.file_path}, 类型: {document.file_type}")
        return FileResponse(document.file_path, media_type=document.file_type)
    else:
        print(f"不支持的文件类型，无法直接查看: {document.file_path}")
        raise HTTPException(status_code=400, detail="不支持的文件类型，无法直接查看")

@app.get("/monitoring/", response_class=HTMLResponse)
async def equipment_monitoring(request: Request):
    return templates.TemplateResponse("equipment_monitoring.html", {"request": request})

# 设备类型维护周期映射表 (月)
MAINTENANCE_CYCLE_MAP = {
    'DCS': 60,
    'PLC': 12,
    '变送器': 1,
    '热电阻': 12,
    '热电偶': 24,
    '调节阀': 3,
    '切断阀': 6,
    '物位计': 3,
    '压力表': 3,
    '双金属温度计': 6
}

@app.get("/monitoring/data/")
async def equipment_monitoring_data(db: Session = Depends(get_db)):
    # 获取所有设备
    equipments = db.query(EquipmentDB).all()
    
    # 获取所有维护记录
    maintenances = db.query(MaintenanceDB).all()
    
    # 准备设备详情数据
    equipment_details = []
    for eq in equipments:
        # 计算已使用时间(月)
        if eq.installation_date:
            used_months = (datetime.now() - eq.installation_date).days / 30.0
        else:
            used_months = 0
        
        # 计算剩余寿命(月)
        remaining_life = eq.working_life - used_months if eq.working_life else 0
        
        # 获取维护周期
        maintenance_cycle = MAINTENANCE_CYCLE_MAP.get(eq.equipment_type, 0)
        
        equipment_details.append({
            "id": eq.id,
            "name": eq.name,
            "model": eq.model,
            "working_life": eq.working_life if eq.working_life is not None else 0,
            "maintenance_cycle": maintenance_cycle,
            "used_months": used_months,
            "remaining_life": remaining_life,
            "next_maintenance": eq.next_maintenance.strftime('%Y-%m-%d') if eq.next_maintenance else None,
            "status": eq.status
        })
    
    # 准备维护历史数据（近6个月）
    maintenance_history = []
    today = date.today()
    for i in range(6):
        month_date = date(today.year, today.month - i, 1) if today.month - i > 0 else date(today.year - 1, 12 + (today.month - i), 1)
        next_month = date(month_date.year, month_date.month + 1, 1) if month_date.month < 12 else date(month_date.year + 1, 1, 1)
        count = len([m for m in maintenances if month_date <= m.maintenance_date.date() < next_month])
        month_name = month_date.strftime("%Y-%m")
        maintenance_history.append({"month": month_name, "count": count})
    
    # 按月份排序
    maintenance_history.sort(key=lambda x: x["month"])
    
    # 设备类型分布
    equipment_type_distribution = {}
    for eq in equipments:
        eq_type = eq.equipment_type or "未分类"
        equipment_type_distribution[eq_type] = equipment_type_distribution.get(eq_type, 0) + 1
    
    return JSONResponse(content={
        "equipment_details": equipment_details,
        "maintenance_history": maintenance_history,
        "equipment_type_distribution": equipment_type_distribution
    })

# 创建数据库表
Base.metadata.create_all(bind=engine)

# 运行应用
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5173)