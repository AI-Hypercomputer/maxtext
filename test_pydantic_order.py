from pydantic import BaseModel, Field

class A(BaseModel):
    a: int = Field(1)

class B(BaseModel):
    b: int = Field(2)

class C(BaseModel):
    c: int = Field(3)

class Config(A, B, C):
    d: int = Field(4)

print(list(Config.model_fields.keys()))
