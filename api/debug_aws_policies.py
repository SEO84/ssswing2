#!/usr/bin/env python3
"""
AWS IAM 정책 상세 분석 스크립트
"""

import boto3
from dotenv import load_dotenv
import os
import json

# 환경변수 로드
load_dotenv()

# STS 클라이언트로 현재 사용자 정보 확인
sts = boto3.client('sts', 
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION'))

try:
    caller_identity = sts.get_caller_identity()
    print(f"=== 현재 AWS 사용자 정보 ===")
    print(f"UserId: {caller_identity.get('UserId')}")
    print(f"Account: {caller_identity.get('Account')}")
    print(f"Arn: {caller_identity.get('Arn')}")
    print()
except Exception as e:
    print(f"사용자 정보 조회 실패: {e}")
    exit(1)

# IAM 클라이언트 생성
iam = boto3.client('iam',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION'))

try:
    # seo 사용자의 직접 연결된 정책 확인
    print("=== seo 사용자 직접 연결된 정책 ===")
    response = iam.list_attached_user_policies(UserName='seo')
    
    for policy in response['AttachedPolicies']:
        policy_arn = policy['PolicyArn']
        policy_name = policy['PolicyName']
        
        print(f"\n📋 정책: {policy_name}")
        print(f"   ARN: {policy_arn}")
        
        # 정책 내용 가져오기
        try:
            if policy_arn.startswith('arn:aws:iam::aws:policy/'):
                # AWS 관리형 정책
                policy_response = iam.get_policy(PolicyArn=policy_arn)
                version_id = policy_response['Policy']['DefaultVersionId']
                version_response = iam.get_policy_version(PolicyArn=policy_arn, VersionId=version_id)
                policy_doc = version_response['PolicyVersion']['Document']
            else:
                # 사용자 정의 정책
                policy_response = iam.get_policy(PolicyArn=policy_arn)
                version_id = policy_response['Policy']['DefaultVersionId']
                version_response = iam.get_policy_version(PolicyArn=policy_arn, VersionId=version_id)
                policy_doc = version_response['PolicyVersion']['Document']
            
            # Deny 규칙이 있는지 확인
            statements = policy_doc.get('Statement', [])
            for i, statement in enumerate(statements):
                if statement.get('Effect') == 'Deny':
                    print(f"   ⚠️  DENY 규칙 발견 (Statement {i}):")
                    print(f"      Action: {statement.get('Action', 'N/A')}")
                    print(f"      Resource: {statement.get('Resource', 'N/A')}")
                    if 'Condition' in statement:
                        print(f"      Condition: {statement['Condition']}")
                        
        except Exception as e:
            print(f"   ❌ 정책 내용 조회 실패: {e}")
    
    # 사용자가 속한 그룹 확인
    print(f"\n=== seo 사용자가 속한 그룹 ===")
    groups_response = iam.get_groups_for_user(UserName='seo')
    
    if groups_response['Groups']:
        for group in groups_response['Groups']:
            group_name = group['GroupName']
            print(f"\n👥 그룹: {group_name}")
            
            # 그룹에 연결된 정책 확인
            group_policies = iam.list_attached_group_policies(GroupName=group_name)
            
            for policy in group_policies['AttachedPolicies']:
                policy_arn = policy['PolicyArn']
                policy_name = policy['PolicyName']
                
                print(f"   📋 그룹 정책: {policy_name}")
                
                # 정책 내용 확인
                try:
                    if policy_arn.startswith('arn:aws:iam::aws:policy/'):
                        policy_response = iam.get_policy(PolicyArn=policy_arn)
                        version_id = policy_response['Policy']['DefaultVersionId']
                        version_response = iam.get_policy_version(PolicyArn=policy_arn, VersionId=version_id)
                        policy_doc = version_response['PolicyVersion']['Document']
                    else:
                        policy_response = iam.get_policy(PolicyArn=policy_arn)
                        version_id = policy_response['Policy']['DefaultVersionId']
                        version_response = iam.get_policy_version(PolicyArn=policy_arn, VersionId=version_id)
                        policy_doc = version_response['PolicyVersion']['Document']
                    
                    # Deny 규칙 확인
                    statements = policy_doc.get('Statement', [])
                    for i, statement in enumerate(statements):
                        if statement.get('Effect') == 'Deny':
                            print(f"   ⚠️  DENY 규칙 발견 (Statement {i}):")
                            print(f"      Action: {statement.get('Action', 'N/A')}")
                            print(f"      Resource: {statement.get('Resource', 'N/A')}")
                            if 'Condition' in statement:
                                print(f"      Condition: {statement['Condition']}")
                                
                except Exception as e:
                    print(f"   ❌ 그룹 정책 내용 조회 실패: {e}")
    else:
        print("그룹에 속하지 않음")
        
except Exception as e:
    print(f"IAM 정책 조회 실패: {e}")
