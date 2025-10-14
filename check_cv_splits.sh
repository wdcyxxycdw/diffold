#!/bin/bash
# 检查CD-HIT交叉验证划分的数据完整性脚本
# 用法: ./check_cv_splits.sh [list_directory_path]

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认路径，如果没有提供参数
LIST_DIR=${1:-"processed_data/cv_splits_cdhit/list"}

echo -e "${BLUE}=== CD-HIT CV split integrity check ===${NC}"
echo -e "Directory: ${LIST_DIR}"
echo ""

# 检查目录是否存在
if [ ! -d "$LIST_DIR" ]; then
    echo -e "${RED}❌ Error: directory $LIST_DIR does not exist${NC}"
    exit 1
fi

# 进入目录
cd "$LIST_DIR" || exit 1

# 检查文件是否存在
missing_files=()
for i in {0..9}; do
    if [ ! -f "fold-${i}_train_ids" ]; then
        missing_files+=("fold-${i}_train_ids")
    fi
    if [ ! -f "valid_fold-${i}" ]; then
        missing_files+=("valid_fold-${i}")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo -e "${RED}❌ Missing files:${NC}"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    exit 1
fi

echo -e "${GREEN}✓ All required files exist${NC}"
echo ""

# 1. Check overlaps between train and valid
echo -e "${BLUE}1. Check overlaps between train and valid:${NC}"
echo "----------------------------------------"
total_overlaps=0
for i in {0..9}; do
    overlap_count=$(comm -12 <(sort "fold-${i}_train_ids") <(sort "valid_fold-${i}") | wc -l)
    if [ "$overlap_count" -eq 0 ]; then
        echo -e "  fold-$i: ${GREEN}✓ no overlap${NC}"
    else
        echo -e "  fold-$i: ${RED}❌ found $overlap_count overlapping sequences${NC}"
        total_overlaps=$((total_overlaps + overlap_count))
    fi
done

if [ "$total_overlaps" -eq 0 ]; then
    echo -e "${GREEN}✓ No overlaps between train and valid in any fold${NC}"
else
    echo -e "${RED}❌ total of $total_overlaps overlapping sequences found${NC}"
fi
echo ""

# 2. Check data integrity
echo -e "${BLUE}2. Check data integrity:${NC}"
echo "----------------------------------------"

# 计算每个fold的总数
fold_totals=()
for i in {0..9}; do
    train_count=$(wc -l < "fold-${i}_train_ids")
    val_count=$(wc -l < "valid_fold-${i}")
    total=$((train_count + val_count))
    fold_totals+=($total)
    echo "  fold-$i: train $train_count + valid $val_count = $total"
done

# 检查所有fold是否有相同的总数
first_total=${fold_totals[0]}
all_same=true
for total in "${fold_totals[@]}"; do
    if [ "$total" -ne "$first_total" ]; then
        all_same=false
        break
    fi
done

echo ""
if [ "$all_same" = true ]; then
    echo -e "${GREEN}✓ Same total size across folds: $first_total${NC}"
else
    echo -e "${RED}❌ Inconsistent total size across folds${NC}"
fi

# 3. Check duplicates across validation sets
echo ""
echo -e "${BLUE}3. Check duplicates across validation sets:${NC}"
echo "----------------------------------------"

# 合并所有验证集
all_val_sequences=$(cat valid_fold-* | sort)
unique_val_sequences=$(echo "$all_val_sequences" | sort -u)

all_val_count=$(echo "$all_val_sequences" | wc -l)
unique_val_count=$(echo "$unique_val_sequences" | wc -l)

if [ "$all_val_count" -eq "$unique_val_count" ]; then
    echo -e "${GREEN}✓ No duplicates across validation sets${NC}"
else
    duplicate_count=$((all_val_count - unique_val_count))
    echo -e "${RED}❌ $duplicate_count duplicate sequences across validation sets${NC}"
    
    # Show duplicate sequences
    echo "Duplicate sequences:"
    echo "$all_val_sequences" | sort | uniq -d | head -10
    if [ $duplicate_count -gt 10 ]; then
        echo "... ($((duplicate_count - 10)) more duplicates)"
    fi
fi

# 4. Summary
echo ""
echo -e "${BLUE}4. Summary:${NC}"
echo "----------------------------------------"
echo "  total folds: 10"
echo "  samples per fold: $first_total"
echo "  total unique validation sequences: $unique_val_count"
echo "  average validation size: $((unique_val_count / 10))"

# 计算验证集大小分布
echo ""
echo "Validation set size per fold:"
for i in {0..9}; do
    val_count=$(wc -l < "valid_fold-${i}")
    echo "  fold-$i: $val_count sequences"
done

echo ""
if [ "$total_overlaps" -eq 0 ] && [ "$all_same" = true ] && [ "$all_val_count" -eq "$unique_val_count" ]; then
    echo -e "${GREEN}🎉 All checks passed! Splits look consistent.${NC}"
else
    echo -e "${RED}⚠️  Issues found. Please inspect the splits.${NC}"
fi
